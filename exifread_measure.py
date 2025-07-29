import exifread

with open("inputs/IMG_0086.jpeg", "rb") as f:
    tags = exifread.process_file(f)

# タグ一覧を表示して、"Depth"や"Auxiliary"など、深度に関連するカスタムタグがあるか確認
for tag in tags:
    print(tag, tags[tag])
