cd $1
zip -r -q $1.zip *.txt
mv *.zip ../../15_map
rm -f *.txt
cd ../../15_map
python script.py -g=gt.zip -s=$1.zip
