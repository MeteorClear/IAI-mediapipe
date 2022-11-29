import cx_Oracle
import os
import datetime


# 환경변수 설정
LOCATION = r"C:\\instantclient_21_7"
os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"]


# 시스템 시작 셋업
print("SYSTEM LOG :: system start..")
print("SYSTEM LOG :: database connection start..")
try:
    dsn = cx_Oracle.makedsn("localhost",1521,'xe')
    db = cx_Oracle.connect("dbp2022",'1234',dsn)
    cursor = db.cursor()

    print("SYSTEM LOG :: db connection success")
except:
    print("SYSTEM LOG :: db connection fail, system terminate")
    exit()

#cursor.execute("""select * from buy_info""")
print("SYSTEM LOG :: system setup start..")

LOGIN_USER = None
G_ADMIN_ID = "Admin01"

print("SYSTEM LOG :: system setup complite")





# 로그인
def user_login():
    global LOGIN_USER
    temp_id = input("id : ")
    temp_pw = input("pw : ")
    login_sql = f"select customer_pw from CUSTOMER where customer_id = '{temp_id}'"
    #print(login_sql)
    try:
        cursor.execute(login_sql)
        loginsql_res = cursor.fetchall()
        v_pw = loginsql_res[0][0]
        #print(loginsql_res, type(loginsql_res))
        #print(loginsql_res[0][0])
    except:
        print("\nLogin fail ::\n\t unkwon id")
        return False
    
    if temp_pw != v_pw:
        print("\nLogin fail ::\n\t no match id and pw")
        return False
    LOGIN_USER = temp_id
    print("\nLogin success ::\n\t Now user:", LOGIN_USER)
    return True


# 로그아웃
def user_logout():
    global LOGIN_USER
    LOGIN_USER = None
    print("Logout success")
    return


# 상품검색
def search_product():
    print("choose way to search product")
    print("\t1. name\n\t2. category\n\t3. manufacturer")
    print("\t0. exit")
    search_mode = input()
    if len(search_mode) == 0:
        return search_product()
    if search_mode == "exit" or search_mode == "0":
        return
    elif search_mode == "name" or search_mode == "1":
        search_product_name()
    elif search_mode == "category" or search_mode == "2":
        search_product_category()
    elif search_mode == "manufacturer" or search_mode == "3":
        search_product_manufacturer()
    else:
        print("unkown command")
        return search_product()
    return

# 이름으로 상품검색
def search_product_name():
    print("type name with whitespace for search:")
    name_token = list(input().split())
    temp_first = True
    #print(name_token)
    if len(name_token) == 0:
        return
    
    search_name_sql = "select product_category, product_name, product_price from product where"
    for token in name_token:
        if not temp_first:
            search_name_sql += " or"
            
        search_name_sql += f" product_name like '%{token}%'"
        temp_first = False
    #print(search_name_sql)

    try:
        cursor.execute(search_name_sql)
        pns_sql_res = cursor.fetchall()
        pns_sql_res = list(pns_sql_res)
    except:
        print("\nERROR 001")
        return

    print("\nsearch result :",len(pns_sql_res))
    print("idx. Category    ::    Product name    ::    Price")
    temp_idx = 0
    for i in pns_sql_res:
        print("%3d. " % temp_idx)
        print("  ::  ".join(list(map(str,i))))
        temp_idx += 1
    return


# 카테고리로 상품 검색
def search_product_category():
    print("Now category")
    category_sql = "select distinct product_category from product"
    try:
        cursor.execute(category_sql)
        ctg_sql_res = cursor.fetchall()
        ctg_sql_res = list(ctg_sql_res)
        for i in ctg_sql_res:
            print("\t",i[0])
    except:
        print("\nERROR 001")
        return

    print("type category:")
    ctg = input()
    search_category_sql = f"select product_category, product_name, product_price from product where product_category = '{ctg}'"
    try:
        cursor.execute(search_category_sql)
        pcs_sql_res = cursor.fetchall()
        pcs_sql_res = list(pcs_sql_res)    
    except:
        print("\nunkown category")
        return

    print("\nsearch result :",len(pcs_sql_res))
    print("idx. Category    ::    Product name    ::    Price")
    temp_idx = 0
    for i in pcs_sql_res:
        print("%3d. " % temp_idx)
        print("  ::  ".join(list(map(str,i))))
        temp_idx += 1
    return


# 제조사로 상품 검색
def search_product_manufacturer():
    print("Now manufacturer")
    manufacturer_sql = "select distinct manufacturer_name from MANUFACTURER"
    try:
        cursor.execute(manufacturer_sql)
        mnf_sql_res = cursor.fetchall()
        mnf_sql_res = list(mnf_sql_res)
        for i in mnf_sql_res:
            print("\t",i[0])
    except:
        print("\nERROR 001")
        return

    print("type manufacturer:")
    mnf = input()
    search_manufacturer_sql = f"select product_category, product_name, product_price from product where manufacturer_id in (select manufacturer_id from manufacturer where manufacturer_name = '{mnf}')"
    #print(search_manufacturer_sql)
    try:
        cursor.execute(search_manufacturer_sql)
        pms_sql_res = cursor.fetchall()
        pms_sql_res = list(pms_sql_res)    
    except:
        print("\nunkown manufacturer")
        return

    print("\nsearch result :",len(pms_sql_res))
    print("idx. Category    ::    Product name    ::    Price")
    temp_idx = 0
    for i in pms_sql_res:
        print("%3d. " % temp_idx)
        print("  ::  ".join(list(map(str,i))))
        temp_idx += 1
    return


def product_supply():
    print("input product name")
    ps_name = input()
    ps_pnum = 0
    new_product = False
    try:
        ps_pnum, new_product = get_product_number(ps_name)
    except:
        print("ERROR 03")
        return
    
    ps_mnum = 0
    if new_product:
        print("input product manufacturer")
        ps_mnf = input()
        
        new_manufacturer = False
        try:
            ps_mnum, new_manufacturer = get_manufacturer_id(ps_mnf)
        except:
            print("ERROR 03")
            return
        if new_manufacturer:
            print("manufacturer register first")
            return
    else:
        ps_m_sql = f"select manufacturer_id from product where product_name = '{ps_name}'"
        try:
            cursor.execute(ps_m_sql)
            ps_mnum = cursor.fetchall()
            ps_mnum = int(list(ps_mnum)[0][0])
            print(ps_mnum)
        except:
            print("ERROR 03")
            return

    print("input supply amount")
    ps_amnt = 0
    try:
        ps_amnt = int(input())
    except:
        print("ERROR 03")
        return

    now_date = datetime.datetime.now().date().strftime("%Y/%m/%d")
    now_date = str(now_date)
    print(now_date)

    idx_sql = "select max(idx) from supply"
    ps_idx = 0
    try:
        cursor.execute(idx_sql)
        ps_idx = cursor.fetchall()
        ps_idx = int(list(ps_idx)[0][0]) + 1
        print(ps_idx)
    except:
        print("ERROR 04")

    # 상품 정보 입력, 기존->수량변경, 신규->등록
    if new_product:
        product_insert_sql = f"insert into product (product_number, product_name, product_stock, manufacturer_id, admin_id) values ({ps_pnum}, '{ps_name}', {ps_amnt}, {ps_mnum}, '{G_ADMIN_ID}')"
        print(product_insert_sql)
        try:
            cursor.execute(product_insert_sql)
        except:
            print("ERROR 04")
            return
    else:
        product_update_sql = f"update product set product_stock = product_stock+{ps_amnt} where product_number = {ps_pnum}"
        print(product_update_sql)
        try:
            cursor.execute(product_update_sql)
        except:
            print("ERROR 04")
            return
    
    supply_insert_sql = f"insert into supply (idx, supply_amount, product_number, manufacturer_id) values ({ps_idx}, {ps_amnt}, {ps_pnum}, {ps_mnum})"
    print(supply_insert_sql)
    #cursor.execute(supply_insert_sql)
    try:
        cursor.execute(supply_insert_sql)
    except:
        print("ERROR 05")
        return

    return

def get_product_number(ps_name):
    ps_n_sql = f"select product_number from product where product_name = '{ps_name}'"
    ps_pnum = 0
    new_product = False
    try:
        cursor.execute(ps_n_sql)
        ps_pnum = cursor.fetchall()
        ps_pnum = int(list(ps_pnum)[0][0])
        print(ps_pnum)
    except:
        print("no target")
        ps_n_sql = "select max(product_number) from product"
        try:
            cursor.execute(ps_n_sql)
            ps_pnum = cursor.fetchall()
            ps_pnum = int(list(ps_pnum)[0][0]) + 1
            new_product = True
            print(ps_pnum)
        except:
            print("ERROR 02")
            return False
    return (ps_pnum, new_product)

def get_manufacturer_id(ps_mnf):
    ps_mnum = 0
    new_manufacturer = False
    ps_m_sql = f"select manufacturer_id from manufacturer where manufacturer_name = '{ps_mnf}'"
    try:
        cursor.execute(ps_m_sql)
        ps_mnum = cursor.fetchall()
        ps_mnum = int(list(ps_mnum)[0][0])
        print(ps_mnum)
    except:
        print("no target")
        ps_m_sql = "select max(manufacturer_id) from manufacturer"
        try:
            cursor.execute(ps_m_sql)
            ps_mnum = cursor.fetchall()
            ps_mnum = int(list(ps_mnum)[0][0]) + 1
            new_manufacturer = True
            print(ps_mnum)
        except:
            print("ERROR 02")
            return False
    return (ps_mnum, new_manufacturer)




# 메인 시스템
while 1:
    print("\n")
    print(f"Now user :: [ {LOGIN_USER} ]")
    print("write command:")

    mode = list(input().split())
    if len(mode)==0:
        continue

    if mode[0] == "exit" or mode[0] == "end" or mode[0] == "quit":
        print("\nDo you really want to quit?\n[y] / n")
        exit_temp = input()
        if exit_temp == "n" or exit_temp == "N" or exit_temp == "NO" or exit_temp == "no" or exit_temp == "No":
            continue
        print("SYSTEM LOG :: system terminate")
        break

    elif mode[0] == "login":
        print()
        if LOGIN_USER != None:
            user_logout()
        user_login()

    elif mode[0] == "logout":
        print()
        user_logout()

    elif mode[0] == "search":
        print()
        search_product()

    elif mode[0] == "supply":
        print()
        product_supply()


    else:
        print()
        print("unkown command")
