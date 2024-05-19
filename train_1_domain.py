import gpl

model_name = 'Snowflake/snowflake-arctic-embed-xs'
# model_name = 'GPL/msmarco-distilbert-margin-mse',
# model_name = 'distilbert-base-uncased'
batch_size = 128
# gpl_steps = 140000
gpl_steps = 10000
output_dir = './output/model_domain'
evaluation_output = f"{output_dir}_evaluation"

dataset = 'dataset'
gpl.train(
    path_to_generated_data=f"generated/{dataset}",
    base_ckpt=model_name,
    # The starting checkpoint of the experiments in the paper
    gpl_score_function="dot",
    # Note that GPL uses MarginMSE loss, which works with dot-product
    batch_size_gpl=batch_size,
    gpl_steps=gpl_steps,

    # Resize the corpus to `new_size` (|corpus|) if needed.
    # When set to None (by default), the |corpus| will be the full size.
    # When set to -1, the |corpus| will be set automatically:
    #   If QPP * |corpus| <= 250K, |corpus| will be the full size;
    #   else QPP will be set 3 and |corpus| will be set to 250K / 3
    new_size=-1,

    # Number of Queries Per Passage (QPP) in the query generation step.
    # When set to -1 (by default), the QPP will be chosen automatically:
    #   If QPP * |corpus| <= 250K, then QPP will be set to 250K / |corpus|;
    #   else QPP will be set 3 and |corpus| will be set to 250K / 3
    queries_per_passage=25,

    output_dir=output_dir,
    evaluation_data=f"./{dataset}",
    evaluation_output=evaluation_output,
    generator="BeIR/query-gen-msmarco-t5-base-v1",
    retrievers=["msmarco-distilbert-base-v3", "msmarco-MiniLM-L-6-v3"],
    # Note that these two retriever model work with cosine-similarity
    retriever_score_functions=["cos_sim", "cos_sim"],
    cross_encoder="cross-encoder/ms-marco-MiniLM-L-6-v2",
    # This prefix will appear as part of the (folder/file) names for query-generation results:
    #   For example, we will have "qgen-qrels/" and "qgen-queries.jsonl" by default.
    qgen_prefix="qgen",
    do_evaluation=True,
    # One can use this flag for enabling the efficient float16 precision
    # use_amp=True
)
