"""
该 .py 文件为 ./test.py 文件的辅助文件, 用于将内容图像、风格图像、掩膜图像、风格化图像整合到一个 .html 文件中.
"""
from os import path
# 创建HTML文件并插入表格和图像
def generate_html(A_list, B_list, trs_full_list, trs_list, config):
    table_list = []
    # print(config.style_dir.split('/')[7:])
    style_dir = path.join('../../../../', *config.style_dir.split('/')[7:])
    # print(style_dir)
    for i, pth in enumerate(A_list):
        A_path = './' + A_list[i].split('/')[-1]

        mask_name = B_list[i].split('/')[-1].split('_')[2] + '_mask_' + '_'.join(B_list[i].split('/')[-1].split('_')[4:])
        # print(mask_name)
        mask_path = path.join(style_dir, mask_name)
        # print(mask_path)

        B_path = './' + B_list[i].split('/')[-1]
        trs_full_path = './' + trs_full_list[i].split('/')[-1]
        trs_path = './' + trs_list[i].split('/')[-1]

        table_list.append(f'<tr> <td>{i+1}</td><td><img src="{A_path}" alt="内容图像{i}"></td> <td><img src="{B_path}" alt="风格图像{i}"></td> <td><img src="{trs_full_path}" alt="生成完整图像{i}"></td><td><img src="{trs_path}" alt="生成图像(with alpha){i}"></td> </tr>')
    html_content_start = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>{'_'.join([config.ckpt_name.split('.')[0].split('_')[2], config.ckpt_name.split('.')[0].split('_')[4], str(config.test_style_size)])}</title>
        <style>
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            td {{
                border: 1px solid black;
                text-align: center;
            }}
            img {{
                width: 100px; /* 可以根据需要调整图像宽度 */
                height: auto;
            }}
        </style>
    </head>
    <body>
        <table>
            <thead><th>序号</th><th>内容图像</th><th>风格图像</th><th>掩膜图像(白色为主体部分)</th><th>风格化图像</th></thead>
            <tbody>
    """
    for i,sentence in enumerate(table_list):
        html_content_start += 4*'\t' + table_list[i] +'\n'
    html_content_last = """
            </tbody>
        </table>
    </body>
    </html>
    """
    html_content_start += html_content_last
    html_content = html_content_start

    # 将HTML内容写入文件
    file_dir = path.join(*A_list[0].split('/')[:-1])
    print(file_dir)
    file_name = 'result_img_table.html'
    file_path = path.join(file_dir, file_name)
    # print(file_path)
    with open('/'+file_path, "w", encoding="utf-8") as file:
        file.write(html_content)

    print(f"文件 '{'/' + file_path}' 已创建。")

if __name__ == "__main__":
    A_list = ['./output/dunhuang_white_main/256/000000000016_content_cropped_image_1_1.jpg','./output/dunhuang_white_main/256/000000000016_content_cropped_image_2_0.jpg']
    generate_html(A_list,A_list,A_list)
    # with open('../imgs/masks/main/cropped_mask_1_1.jpg', 'r'):
    #     print("hello")