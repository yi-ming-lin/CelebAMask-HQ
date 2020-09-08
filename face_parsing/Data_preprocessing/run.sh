WORK_DIR="../../../../../dataset/CelebAMask-HQ/"
python g_mask.py ${WORK_DIR}
python g_partition.py ${WORK_DIR}
python g_color.py ${WORK_DIR}
echo "Finished in $WORK_DIR"

