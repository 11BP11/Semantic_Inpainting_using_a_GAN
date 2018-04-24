

#1: All possibilities
#standard_vals = ['','','','','','','','']
#alt_vals = ['--img_height=16','--use_labels','--lambda_loss=10','--split_data','--gen_use_img','--use_border','--z_dim=500','--drop_discriminator']


#2: Compare interesting ones:
standard_vals = ['','','']
alt_vals = ['--gen_use_img','--z_dim=4000','--drop_discriminator']


command = 'python main.py --dataset celebA --input_height=108 --crop '
#command += '--batch_size=16 '
command += '--epoch=10 '
#command += '--train_size=1000 '
command += '--train '     

      
with open('run_all_celebA_commands.cmd', "w") as f: 
  f.write("echo run_all_celebA_commands started" + "\n")
  f.write("\n" + "timeout 5")
  
  line = "\n" + command + " ".join(standard_vals)
  f.write(line)
    
  name = "Version_basis"
  f.write("\n" + "python save_old_version.py \"" + name + "\"" + "\n")
 
  for i in range(len(standard_vals)):
    vals = standard_vals[:]
    vals[i] =alt_vals[i]
    
    name = ("Version" + str(i) + "_" + alt_vals[i]).replace("-","").replace("=","")
    
    f.write("\n\n" + "echo Starting " + name)
    f.write("\n" + "timeout 20")
    
    line = "\n" + command + " ".join(vals)
    f.write(line)
    
    name = "Version" + str(i) + "_" + alt_vals[i]
    f.write("\n" + "python save_old_version.py \"" + name + "\"")
