读取config文件，注意options sections _sections三个函数的用法

```
import configparser
import os

CONFIG_FILE = 'global.cfg'

if __name__ == '__main__':
    if os.path.exists(os.path.join(os.getcwd(), CONFIG_FILE)):
        config = configparser.ConfigParser()
        config.read(CONFIG_FILE)

        host = config.get("DB_Config", "DATABASE_HOST")
        port = config.get("DB_Config", "DATABASE_PORT")
        name = config.get("DB_Config", "DATABASE_NAME")

        print(host, port, name)

        opt = config.options("DB_Config")
        print(opt)

        sec = config.sections()
        print(sec)
        secs = config._sections['DB_Config']
        print(secs)
        # DB_CON = sec['DB_Config']
        # print(DB_CON)
        # FL_CON = sec['FL_Config']
        # print(FL_CON)
```

写config文件

```
# -* - coding: UTF-8 -* -
import os
import configparser

CONFIG_FILE = "global.cfg"

host = "127.0.0.1"

port = "5432"

name = "DATABASE_NAME"

username = "postgres"

password = "postgres"

if __name__ == "__main__":

         conf = configparser.ConfigParser()

         cfgfile = open(CONFIG_FILE,'w')

         conf.add_section("DB_Config") # 在配置文件中增加一个段

         # 第一个参数是段名，第二个参数是选项名，第三个参数是选项对应的值

         conf.set("DB_Config", "DATABASE_HOST", host)

         conf.set("DB_Config", "DATABASE_PORT", port)

         conf.set("DB_Config", "DATABASE_NAME", name)

         conf.set("DB_Config", "DATABASE_USERNAME", username)

         conf.set("DB_Config", "DATABASE_PASSWORD", password)

         conf.add_section("FL_Config")

         # 将conf对象中的数据写入到文件中

         conf.write(cfgfile)

         cfgfile.close()
```

生成的global.cfg文件

```
[DB_Config]
database_host = 127.0.0.1
database_port = 5432
database_name = DATABASE_NAME
database_username = postgres
database_password = postgres

[FL_Config]

```

