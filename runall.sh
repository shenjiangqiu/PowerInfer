

python scripts/run_sparse_dump_bin.py  --models "ProSparse-llama-7b" \
--datasets wiki,c4,alpaca \
--max-prompts 20 \
--n-predict 1 \
--threads 80 \
--dumpdir ./dumpbins_prosparse_llama_7b_new3 \
--main-bin ./build/bin/main