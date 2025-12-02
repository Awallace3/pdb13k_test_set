#/usr/bin/bash
# get all *.in files
# run sed -n '/molecule/,/units = au/ {/^  [A-Z]/p; /^--/p}' on each file and write results to *.xyz
# use argument to specify the directory
# usage: ./generate_p4_molstrings.sh /path/to/directory
start_dir=$(pwd)
cd $1
for file in *.in; do
    echo $file
    sed -n '/molecule/,/units = au/ {/^  [A-Z]/p; /^--/p}' $file > ${file%.in}.p4str
    echo ${file%.in}.p4str
done
cd $start_dir
