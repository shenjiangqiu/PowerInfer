

python scripts/run_sparse_dump_bin.py  --models "ReluLLaMA-7B" \
--datasets wiki,c4,alpaca \
--max-prompts 20 \
--n-predict 1 \
--threads 64 \
--dumpdir ./dumpbins_ReluLLaMA-7B \
--main-bin ./build/bin/main \
--machine husky5


python scripts/run_sparse_dump_bin.py  --models "Bamboo-7B" \
--datasets wiki,c4,alpaca \
--max-prompts 20 \
--n-predict 1 \
--threads 64 \
--dumpdir ./dumpbins_Bamboo-7B \
--main-bin ./build/bin/main \
--machine husky5


python scripts/run_sparse_dump_bin.py  --models "Bamboo-dpo-7B" \
--datasets wiki,c4,alpaca \
--max-prompts 20 \
--n-predict 1 \
--threads 64 \
--dumpdir ./dumpbins_Bamboo-dpo-7B \
--main-bin ./build/bin/main \
--machine husky5


python scripts/run_sparse_dump_bin.py  --models "ProSparse-llama-7b" \
--datasets wiki,c4,alpaca \
--max-prompts 20 \
--n-predict 1 \
--threads 64 \
--dumpdir ./dumpbins_ProSparse-llama-7b \
--main-bin ./build/bin/main \
--machine husky5