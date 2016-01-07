#!/bin/bash

PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin
#NOTICE: the clean.sh for single file is simple, it is following command:
#grep -e "<doc" -e "</doc>" -v FILENAME >> FILENAME.cln
export PATH
for f_num in $( seq 0 72 )
do
	if [ $f_num -lt 10 ]
	then	grep -e "<doc" -e "</doc>" -v wiki_0$f_num >> wiki_0${f_num}.cln
	else	grep -e "<doc" -e "</doc>" -v wiki_$f_num >> wiki_${f_num}.cln
	fi
done
exit 0
