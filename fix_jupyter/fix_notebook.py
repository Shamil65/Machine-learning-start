import json
import sys

def fix_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb.get('cells', []):
        if 'attachments' in cell:
            # Удаляем все аттачменты с кириллицей или пробелами
            to_remove = [k for k in cell['attachments'] 
                        if any(c in k for c in 'А-Яа-я ') or '%' in k]
            for key in to_remove:
                del cell['attachments'][key]
            # Если аттачменты пустые — убираем ключ
            if not cell['attachments']:
                del cell['attachments']
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print(f"✅ Fixed: {path}")

if __name__ == "__main__":
    fix_notebook(sys.argv[1])