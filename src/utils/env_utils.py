def ensure_package(pkg_name: str):
    """确保包已安装，否则自动安装"""
    import importlib, subprocess, sys
    try:
        importlib.import_module(pkg_name)
    except ImportError:
        print(f"⚙️  Installing missing package: {pkg_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])



def count_element_list_occurrence(list_of_lists):
    '''
    跨列表元素统计，常用于多个列表之间的基因出现的频率统计
    :param list_of_lists: 多个列表的列表
    :return: 返回计数字典
    '''
    from collections import defaultdict
    counter = defaultdict(int)
    for unique_list in list_of_lists:
        for item in set(unique_list):  # 用 set() 保证列表内唯一
            counter[item] += 1
    return dict(counter)

def sanitize_filename(filename):
    import re
    return re.sub(r'[<>:"/\\|?*]', '_', filename)
