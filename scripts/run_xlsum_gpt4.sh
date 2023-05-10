# for lang in "swahili" "tamil" "thai" "russian" "portuguese" "arabic" "hindi" "igbo" "indonesian" "urdu" "kyrgyz" "oromo" "amharic" "azerbaijani" "burmese" "chinese_simplified" "welsh" "kirundi" "hausa" "scottish_gaelic" "nepali" "pashto" "persian" "pidgin" "serbian_cyrillic" "serbian_latin" "sinhala" "somali" "tigrinya" "turkish" "ukrainian" "uzbek" "vietnamese" "yoruba"
# do
#     python -m mega.XLSUM ${lang}
# done

for lang in "turkish"
do
    python -m mega.XLSUM ${lang}
done