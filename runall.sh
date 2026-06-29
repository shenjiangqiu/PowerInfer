python scripts/run_sparse_dump_bin.py  --models Bamboo-dpo-7B \
--datasets wiki,c4,alpaca \
--max-prompts 20 \
--n-predict 1 \
--threads 20 \
--dumpdir ./dumpbins_bamboo_dpo \
--main-bin ./build/bin/main



python scripts/run_sparse_dump_bin.py  --models ReluLLaMA-7B \
--datasets wiki,c4,alpaca \
--max-prompts 20 \
--n-predict 1 \
--threads 20 \
--dumpdir ./dumpbins_relu_llama_7b \
--main-bin ./build/bin/main