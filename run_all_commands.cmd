echo run_all_commands started

timeout 5
python main.py --dataset mnist --input_height=28 --output_height=28 --epoch=10 --train        
python save_old_version.py "Version_basis"


echo Starting Version0_img_height16
timeout 20
python main.py --dataset mnist --input_height=28 --output_height=28 --epoch=10 --train --img_height=16       
python save_old_version.py "Version0_--img_height=16"

echo Starting Version1_use_labels
timeout 20
python main.py --dataset mnist --input_height=28 --output_height=28 --epoch=10 --train  --use_labels      
python save_old_version.py "Version1_--use_labels"

echo Starting Version2_lambda_loss10
timeout 20
python main.py --dataset mnist --input_height=28 --output_height=28 --epoch=10 --train   --lambda_loss=10     
python save_old_version.py "Version2_--lambda_loss=10"

echo Starting Version3_split_data
timeout 20
python main.py --dataset mnist --input_height=28 --output_height=28 --epoch=10 --train    --split_data    
python save_old_version.py "Version3_--split_data"

echo Starting Version4_gen_use_img
timeout 20
python main.py --dataset mnist --input_height=28 --output_height=28 --epoch=10 --train     --gen_use_img   
python save_old_version.py "Version4_--gen_use_img"

echo Starting Version5_use_border
timeout 20
python main.py --dataset mnist --input_height=28 --output_height=28 --epoch=10 --train      --use_border  
python save_old_version.py "Version5_--use_border"

echo Starting Version6_z_dim500
timeout 20
python main.py --dataset mnist --input_height=28 --output_height=28 --epoch=10 --train       --z_dim=500 
python save_old_version.py "Version6_--z_dim=500"

echo Starting Version7_drop_discriminator
timeout 20
python main.py --dataset mnist --input_height=28 --output_height=28 --epoch=10 --train        --drop_discriminator
python save_old_version.py "Version7_--drop_discriminator"