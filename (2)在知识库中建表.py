import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    "host": "localhost",
    "database": "dss知识库",
    "user": "root",
    "password": "123456",
}

TABLES = {
    "Knowledge_Base_Map": """
        CREATE TABLE IF NOT EXISTS Knowledge_Base_Map (
            KnowledgeNo INT PRIMARY KEY COMMENT '知识域编号',
            KnowledgeName VARCHAR(80) NOT NULL COMMENT '知识域名称',
            DomainTag VARCHAR(40) NOT NULL UNIQUE COMMENT '赛道领域标识',
            AudienceSegment VARCHAR(80) NOT NULL COMMENT '主要受众分层',
            BaseWeight DECIMAL(6,3) NOT NULL DEFAULT 1.000 COMMENT '领域基础权重',
            FreshnessHalfLifeDays INT NOT NULL DEFAULT 30 COMMENT '趋势有效半衰期天数',
            Description TEXT COMMENT '知识域说明'
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库领域映射表';
    """,
    "Knowledge_Content_Blocks": """
        CREATE TABLE IF NOT EXISTS Knowledge_Content_Blocks (
            BlockId INT PRIMARY KEY COMMENT '知识块编号',
            KnowledgeNo INT NOT NULL COMMENT '所属知识域编号',
            BlockType VARCHAR(40) NOT NULL COMMENT 'principle/positive_example/negative_example/boundary/action',
            BlockTitle VARCHAR(120) NOT NULL COMMENT '知识块标题',
            ContentText TEXT NOT NULL COMMENT '半结构化自然语言知识内容',
            PlatformScope VARCHAR(40) NOT NULL DEFAULT 'ALL' COMMENT '适用平台或ALL',
            EvidenceLevel VARCHAR(20) NOT NULL DEFAULT 'medium' COMMENT 'high/medium/low',
            SortOrder INT NOT NULL DEFAULT 1 COMMENT '同知识域内展示顺序',
            KeywordsJson JSON DEFAULT NULL COMMENT '召回关键词',
            CONSTRAINT fk_block_knowledge FOREIGN KEY (KnowledgeNo)
                REFERENCES Knowledge_Base_Map(KnowledgeNo) ON DELETE CASCADE ON UPDATE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='半结构化知识块表';
    """,
    "Rule_Definition": """
        CREATE TABLE IF NOT EXISTS Rule_Definition (
            RuleNo INT PRIMARY KEY COMMENT '规则编号',
            KnowledgeNo INT NOT NULL COMMENT '所属知识域编号',
            RuleName VARCHAR(160) NOT NULL COMMENT '规则名称',
            RuleType VARCHAR(40) NOT NULL COMMENT '规则类型：结构/语义/指标/风险/互动',
            RuleCF DECIMAL(6,3) NOT NULL COMMENT '规则先验置信度[-1,1]',
            Priority INT NOT NULL DEFAULT 1 COMMENT '规则优先级',
            AppliesTo VARCHAR(40) NOT NULL DEFAULT 'ALL' COMMENT '适用平台或ALL',
            EvidenceBasis VARCHAR(255) NOT NULL COMMENT '规则依据',
            IsActive BOOLEAN NOT NULL DEFAULT TRUE COMMENT '是否启用',
            CONSTRAINT fk_rule_knowledge FOREIGN KEY (KnowledgeNo)
                REFERENCES Knowledge_Base_Map(KnowledgeNo) ON DELETE CASCADE ON UPDATE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='规则定义表';
    """,
    "Rule_Conditions": """
        CREATE TABLE IF NOT EXISTS Rule_Conditions (
            TermNo INT PRIMARY KEY COMMENT '条件编号',
            RuleNo INT NOT NULL COMMENT '规则编号',
            TermName VARCHAR(80) NOT NULL COMMENT '输入或派生特征名称',
            TermConn VARCHAR(16) NOT NULL COMMENT '比较方式：=,>,<,>=,<=,LIKE,IN,BETWEEN',
            TermKey VARCHAR(255) NOT NULL COMMENT '比较目标值',
            TermWeight DECIMAL(6,3) NOT NULL DEFAULT 1.000 COMMENT '条件在规则内的权重',
            LogicalOp VARCHAR(8) NOT NULL DEFAULT 'AND' COMMENT '与下一条件关系',
            CONSTRAINT fk_condition_rule FOREIGN KEY (RuleNo)
                REFERENCES Rule_Definition(RuleNo) ON DELETE CASCADE ON UPDATE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='规则前件条件表';
    """,
    "Rule_Conclusions": """
        CREATE TABLE IF NOT EXISTS Rule_Conclusions (
            ResultNo INT PRIMARY KEY COMMENT '结论编号',
            RuleNo INT NOT NULL COMMENT '规则编号',
            ResultType VARCHAR(40) NOT NULL COMMENT 'Score_Mod/Text_Advice/Risk_Tag/Feature_Boost',
            ScoreModifier DECIMAL(7,3) NOT NULL DEFAULT 0.000 COMMENT '得分修正值',
            AdviceText TEXT NOT NULL COMMENT '可执行优化建议',
            DerivedFeature VARCHAR(80) DEFAULT NULL COMMENT '可产生的中间特征',
            IsFinal BOOLEAN NOT NULL DEFAULT TRUE COMMENT '是否最终结论',
            CONSTRAINT fk_conclusion_rule FOREIGN KEY (RuleNo)
                REFERENCES Rule_Definition(RuleNo) ON DELETE CASCADE ON UPDATE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='规则后件结论表';
    """,
    "Prompt_Registry": """
        CREATE TABLE IF NOT EXISTS Prompt_Registry (
            PromptNo INT PRIMARY KEY COMMENT '提示编号',
            KnowledgeNo INT NOT NULL COMMENT '关联知识域编号',
            PromptTitle VARCHAR(255) NOT NULL COMMENT '前端追加采集问题',
            PromptType VARCHAR(30) NOT NULL COMMENT 'Input/Dropdown/Checkbox/Slider',
            TargetAttribute VARCHAR(80) NOT NULL COMMENT '写入评分上下文的属性名',
            OptionsJson JSON DEFAULT NULL COMMENT '可选项配置',
            RequiredLevel VARCHAR(20) NOT NULL DEFAULT 'optional' COMMENT 'required/recommended/optional',
            CONSTRAINT fk_prompt_knowledge FOREIGN KEY (KnowledgeNo)
                REFERENCES Knowledge_Base_Map(KnowledgeNo) ON DELETE CASCADE ON UPDATE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='交互提示注册表';
    """,
    "Semantic_Network_Edge": """
        CREATE TABLE IF NOT EXISTS Semantic_Network_Edge (
            EdgeId INT PRIMARY KEY COMMENT '语义边编号',
            KnowledgeNo INT NOT NULL COMMENT '所属知识域编号',
            SourceNode VARCHAR(80) NOT NULL COMMENT '源语义节点',
            TargetNode VARCHAR(80) NOT NULL COMMENT '目标语义节点',
            RelationType VARCHAR(40) NOT NULL COMMENT 'evokes/belongs_to/enhances/risks/opposes',
            EdgeWeight DECIMAL(6,3) NOT NULL COMMENT '关联强度[0,1]',
            ExamplePhrase VARCHAR(180) NOT NULL COMMENT '可解释样例短语',
            CONSTRAINT fk_semantic_knowledge FOREIGN KEY (KnowledgeNo)
                REFERENCES Knowledge_Base_Map(KnowledgeNo) ON DELETE CASCADE ON UPDATE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='语义网络关系表';
    """,
}


def init_database_tables():
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        print(f"✅ 成功连接数据库: {DB_CONFIG['database']}")

        cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
        for table_name in [
            "Rule_Conclusions",
            "Rule_Conditions",
            "Prompt_Registry",
            "Semantic_Network_Edge",
            "Rule_Definition",
            "Knowledge_Content_Blocks",
            "Knowledge_Base_Map",
        ]:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")

        for table_name, ddl in TABLES.items():
            cursor.execute(ddl)
            print(f"📦 已创建表: {table_name}")

        connection.commit()
        print("🎉 知识库表结构已按 DSS 规则链模型重建完成。")

    except Error as e:
        print(f"❌ 数据库操作失败: {e}")
        if connection:
            connection.rollback()
    finally:
        if connection and connection.is_connected():
            if "cursor" in locals():
                cursor.close()
            connection.close()
            print("🔒 数据库连接已关闭。")


if __name__ == "__main__":
    init_database_tables()