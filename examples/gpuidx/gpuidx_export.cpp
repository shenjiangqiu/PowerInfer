#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <fstream>
#include <vector>

static bool export_gpuidx(const std::string & input_path,
                           const std::string & json_path,
                           const std::string & bin_path) {
    // ---- 1. Open GGUF with metadata only (no_alloc) ----
    struct ggml_context * ctx_data = NULL;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_data,
    };
    struct gguf_context * ctx = gguf_init_from_file(input_path.c_str(), params);
    if (!ctx) {
        fprintf(stderr, "error: failed to init gguf from '%s'\n", input_path.c_str());
        return false;
    }

    int n_tensors = gguf_get_n_tensors(ctx);
    size_t data_offset = gguf_get_data_offset(ctx);
    printf("Loaded %s: %d tensors, data offset = %zu\n",
           input_path.c_str(), n_tensors, data_offset);

    // ---- 2. Collect tensor metadata from ggml_tensors ----
    struct tensor_meta {
        std::string name;
        uint32_t type;
        int n_dims;
        int64_t ne[GGML_MAX_DIMS];
        size_t file_offset;  // absolute offset in the source file
        size_t nbytes;
        size_t bin_offset;   // offset in the output binary blob
    };
    std::vector<tensor_meta> metas;

    size_t cur_bin_offset = 0;
    for (int i = 0; i < n_tensors; i++) {
        const char * tname = gguf_get_tensor_name(ctx, i);
        size_t t_off = gguf_get_tensor_offset(ctx, i);

        struct ggml_tensor * cur = ggml_get_tensor(ctx_data, tname);
        if (!cur) {
            fprintf(stderr, "error: tensor '%s' not found in ctx_data\n", tname);
            gguf_free(ctx);
            ggml_free(ctx_data);
            return false;
        }

        int n_dims = cur->n_dims;
        int64_t ne[GGML_MAX_DIMS];
        for (int d = 0; d < GGML_MAX_DIMS; d++) ne[d] = cur->ne[d];

        size_t nbytes = ggml_nbytes(cur);
        size_t file_off = data_offset + t_off;

        metas.push_back({
            tname,
            (uint32_t)cur->type,
            n_dims,
            { ne[0], ne[1], ne[2], ne[3] },
            file_off,
            nbytes,
            cur_bin_offset,
        });

        cur_bin_offset += nbytes;
    }

    // ---- 3. Write JSON metadata ----
    FILE * jf = fopen(json_path.c_str(), "w");
    if (!jf) {
        fprintf(stderr, "error: cannot open JSON output '%s'\n", json_path.c_str());
        gguf_free(ctx);
        ggml_free(ctx_data);
        return false;
    }

    fprintf(jf, "{\n");
    fprintf(jf, "  \"source\": \"%s\",\n", input_path.c_str());

    // vram_capacity kv
    {
        int key_idx = gguf_find_key(ctx, "split.vram_capacity");
        if (key_idx >= 0) {
            uint64_t cap = gguf_get_val_u64(ctx, key_idx);
            fprintf(jf, "  \"vram_capacity\": %" PRIu64 ",\n", cap);
        }
    }

    int n_layers = n_tensors / 2;
    fprintf(jf, "  \"n_layers\": %d,\n", n_layers);
    fprintf(jf, "  \"tensors\": [\n");

    for (int i = 0; i < n_tensors; i++) {
        const tensor_meta & m = metas[i];
        const char * type_name = ggml_type_name((enum ggml_type)m.type);

        fprintf(jf, "    {\n");
        fprintf(jf, "      \"name\": \"%s\",\n", m.name.c_str());
        fprintf(jf, "      \"dtype\": \"%s\",\n", type_name);
        fprintf(jf, "      \"shape\": [");
        for (int d = m.n_dims - 1; d >= 0; d--) {
            fprintf(jf, "%" PRId64 "%s", m.ne[d], (d == 0) ? "" : ", ");
        }
        fprintf(jf, "],\n");
        fprintf(jf, "      \"offset\": %zu,\n", m.bin_offset);
        fprintf(jf, "      \"nbytes\": %zu\n", m.nbytes);
        fprintf(jf, "    }%s\n", (i < n_tensors - 1) ? "," : "");
    }

    fprintf(jf, "  ]\n");
    fprintf(jf, "}\n");
    fclose(jf);
    printf("Wrote JSON metadata to %s\n", json_path.c_str());

    // ---- 4. Copy tensor data from source to binary blob ----
    FILE * sf = fopen(input_path.c_str(), "rb");
    if (!sf) {
        fprintf(stderr, "error: cannot re-open source '%s'\n", input_path.c_str());
        gguf_free(ctx);
        ggml_free(ctx_data);
        return false;
    }

    FILE * bf = fopen(bin_path.c_str(), "wb");
    if (!bf) {
        fprintf(stderr, "error: cannot open BIN output '%s'\n", bin_path.c_str());
        fclose(sf);
        gguf_free(ctx);
        ggml_free(ctx_data);
        return false;
    }

    std::vector<char> buf;
    for (const tensor_meta & m : metas) {
        buf.resize(m.nbytes);
        if (fseek(sf, (long)m.file_offset, SEEK_SET) != 0) {
            fprintf(stderr, "error: fseek to %zu failed\n", m.file_offset);
            fclose(bf); fclose(sf);
            gguf_free(ctx); ggml_free(ctx_data);
            return false;
        }
        if (fread(buf.data(), 1, m.nbytes, sf) != m.nbytes) {
            fprintf(stderr, "error: fread %zu bytes failed for '%s'\n",
                    m.nbytes, m.name.c_str());
            fclose(bf); fclose(sf);
            gguf_free(ctx); ggml_free(ctx_data);
            return false;
        }
        fwrite(buf.data(), 1, m.nbytes, bf);
    }

    fclose(bf);
    fclose(sf);

    printf("Wrote binary data to %s\n", bin_path.c_str());

    // ---- 5. Cleanup ----
    ggml_free(ctx_data);
    gguf_free(ctx);

    printf("Done.\n");
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: %s <path_to.generated.gpuidx> [output_prefix]\n", argv[0]);
        printf("  outputs: <prefix>.gpuidx.json + <prefix>.gpuidx.bin\n");
        return -1;
    }

    std::string input_path(argv[1]);
    std::string output_prefix;
    if (argc >= 3) {
        output_prefix = argv[2];
    } else {
        output_prefix = input_path;
        size_t pos = output_prefix.find(".generated.gpuidx");
        if (pos != std::string::npos) {
            output_prefix = output_prefix.substr(0, pos);
        }
    }

    std::string json_path = output_prefix + ".gpuidx.json";
    std::string bin_path  = output_prefix + ".gpuidx.bin";

    bool ok = export_gpuidx(input_path, json_path, bin_path);
    return ok ? 0 : 1;
}
