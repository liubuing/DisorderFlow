"""PAE 密集批处理任务传输测试：验证分批保持顺序且不丢任务。

守护 scripts.resume_pae_dense_batched.batches：
分块大小不超过 96、拼接后与原任务序列完全一致（顺序敏感）、空输入返回空——
防止分批打乱或丢失任务导致断点续跑覆盖不全。
"""
from scripts.resume_pae_dense_batched import batches


# 传输不变量：分批不得重排（展平后与原序一致）、不得丢任务、块大小受控、
# 空批次安全——防止断点续跑时任务被静默跳过。
def test_transport_partition_keeps_exact_order_and_all_jobs():
    jobs=[{'id':i,'seed':7103+i%3} for i in range(2310)]
    chunks=batches(jobs)
    assert max(map(len,chunks))==96
    assert [j for chunk in chunks for j in chunk]==jobs
    assert batches([])==[]
