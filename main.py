doc_write_path = 'D:\\Dublin_City_University\\CA684\\assignment\\Output\\output.txt'
price_range = 20
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from math import ceil, floor
from PIL import Image
import requests
from io import BytesIO
import re as regular_expression
from contextlib import redirect_stdout
import pandas as pd
text_model = SentenceTransformer('bert-base-german-cased')
cnn_model = SentenceTransformer('clip-ViT-B-32')
german_stop_words = stopwords.words('german')
similarity_output = dict()
offers_training_df = pd.read_parquet('D:\\Dublin_City_University\\CA684\\assignment\\offers_training.parquet')

aboutyou_dataset = offers_training_df.loc[offers_training_df['shop'].isin(['aboutyou'])][['offer_id', 'brand', 'color', 'title', 'description', 'image_urls', 'price']]
zalando_dataset = offers_training_df.loc[offers_training_df['shop'].isin(['zalando'])][['offer_id', 'brand', 'color', 'title', 'description', 'image_urls', 'price']]

brands = offers_training_df['brand'].astype(str).apply(lambda x: x.lower()).unique()


def read_image(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


# about you
au_df = pd.DataFrame()
au_df['offer_id'] = aboutyou_dataset['offer_id']
au_df['title'] = (aboutyou_dataset['title']).astype(str).apply(lambda x: regular_expression.sub('[^a-zA-Z0-9]+', ' ', x).lower())
au_df['title'] = au_df['title'].astype(str).apply(lambda x: " ".join([word for word in x.split() if word not in german_stop_words]))
au_df['color'] = (aboutyou_dataset['color']).astype(str).apply(lambda x: regular_expression.sub('[^a-zA-Z0-9]+', ' ', x).lower())
au_df['color'] = au_df['color'].astype(str).apply(lambda x: ' '.join(set(x.split())))
au_df['image_urls'] = aboutyou_dataset['image_urls']
au_df['price'] = aboutyou_dataset['price']
au_df['brand'] = aboutyou_dataset['brand'].astype(str).apply(lambda x: x.lower())

# au_df['description'] = (aboutyou_dataset['description']).astype(str).apply(lambda x: regular_expression.sub('[^a-zA-Z0-9]+', ' ', x).lower())
# au_df['description'] = au_df['description'].astype(str).apply(lambda x: ' '.join(set(regular_expression.findall('[a-z]{2,}', x))))
# au_df['description'] = au_df['description'].astype(str).apply(lambda x: " ".join([word for word in x.split() if word not in german_stop_words]))

au_df = au_df.reset_index(drop=True)

# zalando
z_df = pd.DataFrame()
z_df['offer_id'] = zalando_dataset['offer_id']
z_df['title'] = (zalando_dataset['title']).astype(str).apply(lambda x: regular_expression.sub('[^a-zA-Z0-9]+', ' ', x).lower())
z_df['title'] = z_df['title'].astype(str).apply(lambda x: " ".join([word for word in x.split() if word not in german_stop_words]))
z_df['color'] = (zalando_dataset['color']).astype(str).apply(lambda x: regular_expression.sub('[^a-zA-Z0-9]+', ' ', x).lower())
z_df['color'] = z_df['color'].astype(str).apply(lambda x: ' '.join(set(x.split())))
z_df['image_urls'] = zalando_dataset['image_urls']
z_df['price'] = zalando_dataset['price']
z_df['brand'] = zalando_dataset['brand'].astype(str).apply(lambda x: x.lower())

# au_df['description'] = (aboutyou_dataset['description']).astype(str).apply(lambda x: regular_expression.sub('[^a-zA-Z0-9]+', ' ', x).lower())
# au_df['description'] = au_df['description'].astype(str).apply(lambda x: ' '.join(set(regular_expression.findall('[a-z]{2,}', x))))
# au_df['description'] = au_df['description'].astype(str).apply(lambda x: " ".join([word for word in x.split() if word not in german_stop_words]))

z_df = z_df.reset_index(drop=True)


for brand in sorted(brands):
    print(brand)
    with open(doc_write_path, 'a') as f:
        with redirect_stdout(f):
            brand_sim = dict()
            segment_au_df = au_df.loc[au_df['brand'] == brand]
            segment_z_df = z_df.loc[z_df['brand'] == brand]
            segment_au_df = segment_au_df.reset_index(drop=True)
            segment_z_df = segment_z_df.reset_index(drop=True)

            aboutyou_title_embeddings = text_model.encode(segment_au_df['title'])
            zalando_title_embeddings = text_model.encode(segment_z_df['title'])

            for zal_index in range(len(zalando_title_embeddings)):
                brand_sim[str(segment_z_df['offer_id'][zal_index])] = {}
                for abtu_index in range(len(aboutyou_title_embeddings)):
                    title_similarity_score = util.cos_sim(zalando_title_embeddings[zal_index], aboutyou_title_embeddings[abtu_index])
                    title_similarity_score = float(title_similarity_score[0][0])
                    if title_similarity_score > 0.70:
                        aboutyou_color_embeddings = text_model.encode(segment_au_df['color'][abtu_index])
                        zalando_color_embeddings = text_model.encode(segment_z_df['color'][zal_index])
                        color_similarity_score = util.cos_sim(zalando_color_embeddings, aboutyou_color_embeddings)
                        color_similarity_score = float(color_similarity_score[0][0])
                        if color_similarity_score > 0.90:
                            if ceil(segment_au_df['price'][abtu_index]) in range(floor(segment_z_df['price'][zal_index]) - price_range, ceil(segment_z_df['price'][zal_index]) + price_range):
                                try:
                                    img_emd_zal = cnn_model.encode([read_image((requests.get(image_item, timeout=5).content)) for image_item in segment_z_df['image_urls'][zal_index]], batch_size=128, convert_to_tensor=True)
                                    img_emb_abtu = cnn_model.encode([read_image((requests.get(image_item, timeout=5).content)) for image_item in segment_au_df['image_urls'][abtu_index]], batch_size=128, convert_to_tensor=True)
                                except Exception as errmsg:
                                    brand_sim[str(segment_z_df['offer_id'][zal_index])].update({segment_au_df['offer_id'][abtu_index]: title_similarity_score})
                                    continue
                                image_similarity_score = util.cos_sim(img_emd_zal, img_emb_abtu)
                                image_similarity_score = float(image_similarity_score[0][0])
                                if image_similarity_score > 0.90:
                                    brand_sim[str(segment_z_df['offer_id'][zal_index])].update({segment_au_df['offer_id'][abtu_index]: image_similarity_score})
                if brand_sim[str(segment_z_df['offer_id'][zal_index])] != {}:
                    about_you_match_offer_id = max(brand_sim[str(segment_z_df['offer_id'][zal_index])], key=brand_sim[str(segment_z_df['offer_id'][zal_index])].get)
                    max_similarity_score = max(brand_sim[str(segment_z_df['offer_id'][zal_index])].values())
                    similarity_output = {str(segment_z_df['offer_id'][zal_index]): {about_you_match_offer_id: max_similarity_score}}
                    print(str(segment_z_df['offer_id'][zal_index]) + " " + about_you_match_offer_id)






