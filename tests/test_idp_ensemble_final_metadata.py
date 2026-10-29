"""IDP 集成最终元数据发现测试：验证记录文本拼接包含标题与实体描述。

守护 scripts.discover_idp_ensemble_final_metadata.record_text：
RCSB entry 的 title 与 polymer entity 描述必须都出现在拼接文本中——
防止元数据检索漏掉实体级字段导致语义筛选漏检。
"""
from scripts.discover_idp_ensemble_final_metadata import record_text


# 文本拼接完整性：标题与实体描述缺一不可——防止实体描述字段被静默丢弃。
def test_record_text_includes_title_and_entity_descriptions():
    text = record_text({
        "title": "anti-tau Fab",
        "polymer_entities": [{"description": "microtubule-associated protein tau"}],
    })
    assert "anti-tau Fab" in text
    assert "microtubule-associated protein tau" in text
