import mysql.connector
from mysql.connector import Error

def connect_to_mysql():
    """连接到 MySQL 数据库并执行简单的测试查询"""
    connection = None
    try:
        # 1. 建立数据库连接
        # 请将下面的参数替换为你自己的数据库信息
        connection = mysql.connector.connect(
            host='localhost',          # 数据库主机地址，如果是本地填 'localhost' 或 '127.0.0.1'
            database='dss知识库',  # 你要连接的具体数据库名称
            user='root',      # 数据库用户名 (例如: 'root')
            password='123456'   # 数据库密码
        )

        # 检查连接是否成功
        if connection.is_connected():
            db_info = connection.get_server_info()
            print(f"✅ 成功连接到 MySQL 数据库。服务器版本: {db_info}")

            # 2. 创建游标对象 (Cursor) 
            # 游标用于执行 SQL 语句并获取结果
            cursor = connection.cursor()

            # 3. 执行 SQL 语句
            cursor.execute("SELECT DATABASE();")
            
            # 4. 获取查询结果
            record = cursor.fetchone()
            print(f"👉 当前连接的数据库是: {record[0]}")

            # --- 你可以在这里添加你的业务逻辑 (例如 INSERT, UPDATE, SELECT 等) ---
            # 如果是执行修改数据的操作 (INSERT/UPDATE/DELETE)，别忘了:
            # connection.commit()

    except Error as e:
        # 捕获并打印连接过程中的任何错误
        print(f"❌ 连接 MySQL 数据库时发生错误: {e}")

    finally:
        # 5. 确保在最后关闭连接释放资源
        if connection is not None and connection.is_connected():
            # 先关闭游标，再关闭连接
            if 'cursor' in locals():
                cursor.close()
            connection.close()
            print("🔒 数据库连接已安全关闭。")

# 运行函数
if __name__ == '__main__':
    connect_to_mysql()