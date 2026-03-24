pip uninstall sageattn3 -y
rm -rf build
python ~/SageAttention-int4-main/sageattention3_blackwell/setup.py install > layout_debug_full.txt 2>&1
grep -nE "PrintType|acc_conversion_flatten|layout|instantiation of|error:|/home/xtzhao/SageAttention-int4-main/sageattention3_blackwell/sageattn3/blackwell/softmax_fused.h" layout_debug_full.txt > layout_debug_key.txt