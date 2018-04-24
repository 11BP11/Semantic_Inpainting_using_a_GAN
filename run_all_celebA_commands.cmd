echo run_all_celebA_commands started

timeout 5
python main.py --dataset celebA --input_height=108 --crop --epoch=8 --train   
python save_old_version.py "Version_basis"


echo Starting Version0_gen_use_img
timeout 20
python main.py --dataset celebA --input_height=108 --crop --epoch=8 --train --gen_use_img  
python save_old_version.py "Version0_--gen_use_img"

echo Starting Version1_z_dim4000
timeout 20
python main.py --dataset celebA --input_height=108 --crop --epoch=8 --train  --z_dim=1000
python save_old_version.py "Version1_--z_dim=4000"

echo Starting Version2_drop_discriminator
timeout 20
python main.py --dataset celebA --input_height=108 --crop --epoch=8 --train   --drop_discriminator
python save_old_version.py "Version2_--drop_discriminator"


echo Starting drop_disc & z_dim
timeout 20
python main.py --dataset celebA --input_height=108 --crop --epoch=8 --train --z_dim=4000 --drop_discriminator
python save_old_version.py "Version2_--drop_discriminator_--z_dim=4000"


echo Starting Version3_--z_dim=2500--img_height=12 (1000 + 24*64*3 = 1000 + 4608 > 2500 + 2304 = 2500 + 12*64*3)
timeout 20
python main.py --dataset celebA --input_height=108 --crop --epoch=8 --train  --z_dim=2500 --img_height=12
python save_old_version.py "Version1_--z_dim=4000_--img_height=12"
