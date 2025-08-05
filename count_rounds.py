import json

def count_rounds(file_path):
    """统计JSON文件中safe和harm rounds的数量"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    safe_count = 0
    harm_count = 0
    unknown_count = 0
    
    # 直接查看顶级的rounds字段
    if 'rounds' in data:
        rounds = data['rounds']
        total_rounds = len(rounds)
        
        print(f"总rounds数: {total_rounds}")
        
        for round_item in rounds:
            category = round_item.get('category', 'unknown')
            if category == 'safe':
                safe_count += 1
            elif category == 'harm':
                harm_count += 1
            else:
                unknown_count += 1
                print(f"未知类别: {category}")
    else:
        print("未找到rounds字段")
        print("可用的顶级字段:", list(data.keys()))
        return
    
    print(f"Safe rounds: {safe_count}")
    print(f"Harm rounds: {harm_count}")
    if unknown_count > 0:
        print(f"未知类别rounds: {unknown_count}")
    print(f"总计: {safe_count + harm_count + unknown_count}")
    
    return safe_count, harm_count

if __name__ == "__main__":
    count_rounds("data/safe_multi_round.json") 