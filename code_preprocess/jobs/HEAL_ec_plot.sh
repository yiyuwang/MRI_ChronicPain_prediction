#!/bin/sh
# Yiyu Wang
# 2023-09-11
# modified from here: https://github.com/Woutervdbos/DTI-Motion/tree/master

# load fsl for $FSLDIR
module load contribs poldrack
module load fsl/6.0.6.2


if [ $# -lt 1 ] ; then
    echo "Usage: `basename $0` <eddy current ecclog file>"
    exit 1;
fi

logfile=$1;
output_dir=$2;
subject=$3;
subject_dir=$output_dir/$subject;

cd $subject_dir;
basenm=`basename $logfile .ecclog`;


nums=`grep -n 'Final' $logfile | sed 's/:.*//'`;

touch grot_ts.txt
touch grot.mat

firsttime=yes;
m=1;
for n in $nums ; do
    echo "Timepoint $m"
    n1=`echo $n + 1 | bc` ;
    n2=`echo $n + 5 | bc` ;
    sed -n  "$n1,${n2}p" $logfile > grot.mat ;
    if [ $firsttime = yes ] ; then firsttime=no; cp grot.mat grot.refmat ; cp grot.mat grot.oldmat ; fi
    absval=`$FSLDIR/bin/rmsdiff grot.mat grot.refmat $basenm`;
    relval=`$FSLDIR/bin/rmsdiff grot.mat grot.oldmat $basenm`;
    cp grot.mat grot.oldmat
    echo $absval $relval >> ec_disp.txt ;
    $FSLDIR/bin/avscale --allparams grot.mat $basenm | grep 'Rotation Angles' | sed 's/.* = //' >> ec_rot.txt ;
    $FSLDIR/bin/avscale --allparams grot.mat $basenm | grep 'Translations' | sed 's/.* = //' >> ec_trans.txt ;
    m=`echo $m + 1 | bc`;
done

echo "absolute" > grot_labels.txt
echo "relative" >> grot_labels.txt

$FSLDIR/bin/fsl_tsplot -i ec_disp.txt -t 'Eddy Current estimated mean displacement (mm)' -l grot_labels.txt -o ec_disp.png

echo "x" > grot_labels.txt
echo "y" >> grot_labels.txt
echo "z" >> grot_labels.txt

$FSLDIR/bin/fsl_tsplot -i ec_rot.txt -t 'Eddy Current estimated rotations (radians)' -l grot_labels.txt -o ec_rot.png
$FSLDIR/bin/fsl_tsplot -i ec_trans.txt -t 'Eddy Current estimated translations (mm)' -l grot_labels.txt -o ec_trans.png

# clean up temp files
/bin/rm grot_labels.txt grot.oldmat grot.refmat grot.mat
