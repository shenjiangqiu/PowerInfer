

python scripts/run_sparse_dump_bin.py  --models "ReluLLaMA-7B" \
--datasets wiki \
--max-prompts 2 \
--n-predict 1 \
--threads 8 \
--dumpdir ./dumpbins \
--main-bin ./build_release/bin/main \
--machine mac