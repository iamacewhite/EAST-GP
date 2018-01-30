echo $1
python eval.py --test_data_path $2 --output_dir $1 --gpu_list $3 --text_scale 576 --base_model resnet_v1_101 --loss dice --checkpoint_path $4 --checkpoint_step $5
cd $1
ls | grep txt | wc -l
zip -r -q $1.zip *.txt
mv *.zip ../../$7_map
cd ..
rm -rf $1
cd ../$7_map
python script.py -g=$6 -s=$1.zip
