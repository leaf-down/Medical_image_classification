from PIL import Image
import os

input_path = r'E:\PETMRI\testGFPPC\PET'
output_path = r'E:\PETMRI\testGFPPC\PET_output'

os.makedirs(output_path, exist_ok=True)


for image_name in os.listdir(input_path):

    if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        continue

    image_path = os.path.join(input_path, image_name)

    try:
        
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')  
            img_ycbcr = img.convert('YCbCr')
            y_channel = img_ycbcr.split()[0]  

            output_name = os.path.splitext(image_name)[0] + '.png'
            output_file = os.path.join(output_path, output_name)

            # 保存为 PNG（无损格式）
            y_channel.save(output_file, format='PNG')
            print(f"processing: {image_name} -> {output_name}")

    except Exception as e:
        print(f"failed {image_name}: {str(e)}")
