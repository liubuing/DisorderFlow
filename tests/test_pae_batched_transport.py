from scripts.resume_pae_dense_batched import batches


def test_transport_partition_keeps_exact_order_and_all_jobs():
    jobs=[{'id':i,'seed':7103+i%3} for i in range(2310)]
    chunks=batches(jobs)
    assert max(map(len,chunks))==96
    assert [j for chunk in chunks for j in chunk]==jobs
    assert batches([])==[]
