from utils import binarify_mask
from utils import get_images

def main():
    image_dir = '/mnt/sda/Dataset/style_image/dunhuang_style'

    input_mask_dir = '/base/mask'
    output_mask_dir = '/enhanced/mask'
    
    input_mask_dir = image_dir + input_mask_dir
    output_mask_dir = image_dir + output_mask_dir
    
    print(input_mask_dir, output_mask_dir)
    input_image_list = get_images.get_data(img_dir=input_mask_dir)
    print(input_image_list)

    for index, p in enumerate(input_image_list):
        print(f"正在处理第 {index+1} 个文件")
        output_image_path_name = output_mask_dir + '/' + input_image_list[index].split('/')[-1]
        print(output_image_path_name)
        binarify_mask.binarify_mask(
            in_image_path_name=input_image_list[index], 
            out_image_path_name=output_image_path_name
            )

if __name__ == '__main__':
    main()