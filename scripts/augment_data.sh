python /home/nampv1/projects/asr/asr_ft/augment_data/generate_addess.py \
--text_path=/media/nampv1/hdd/data/vn_commune_district_province/raw/text/commune_list_with_prefix.txt \
--output_dir=/media/nampv1/hdd/data/Voice-of-Address/commune_list_with_prefix \
--provider gtts \
--delay 3.0 \
--batch_size 10



python /home/nampv1/projects/asr/asr_ft/augment_data/generate_addess.py \
--text_path=/media/nampv1/hdd/data/vn_commune_district_province/raw/text/cdp_list_with_no_prefix.txt \
--output_dir=/media/nampv1/hdd/data/Voice-of-Address/cdp_list_with_no_prefix \
--provider gtts \
--delay 3.0 \
--batch_size 10