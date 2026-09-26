import pandas as pd
import numpy as np
import json


def process_youtube_pipeline(csv_path, json_path, country_name):
    # 1. Load Raw Dataset with Encoding Fallback (For RU, JP, KR, MX)
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin1')

    # Load Category JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        category_data = json.load(f)

    category_mapping = {
        int(item['id']): item['snippet']['title']
        for item in category_data['items']
    }
    df['category_name'] = df['category_id'].map(category_mapping)

    # 2. Missing Values Handling
    df['description'] = df['description'].fillna('Not found')
    df['category_name'] = df['category_name'].fillna('Unknown')
    df['title'] = df['title'].fillna('')

    # 3. Datetime Parsing & Conversion
    df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
    df['publish_time'] = pd.to_datetime(df['publish_time'])

    # 4. Deduplication
    df_unique = df.copy()
    df_unique.drop_duplicates('video_id', keep='first', inplace=True)

    # 5. Feature Engineering
    publish_naive = (
        df_unique['publish_time'].dt.tz_localize(None)
        if df_unique['publish_time'].dt.tz is not None
        else df_unique['publish_time']
    )
    df_unique['days_to_trend'] = (df_unique['trending_date'] - publish_naive).dt.days

    # Text Metrics
    df_unique['title_length'] = df_unique['title'].str.len()
    uppercase_count = df_unique['title'].str.findall(r'[A-Z]').str.len()
    df_unique['title_uppercase_ratio'] = (uppercase_count / df_unique['title_length']).fillna(0)
    df_unique['tags_count'] = df_unique['tags'].fillna('').str.split('|').str.len()

    # Engagement Performance Rates
    df_unique['likes_rate'] = np.where(df_unique['views'] > 0, df_unique['likes'] / df_unique['views'], 0)
    df_unique['comments_rate'] = np.where(df_unique['views'] > 0, df_unique['comment_count'] / df_unique['views'], 0)

    # Time Features
    df_unique['publish_hour'] = df_unique['publish_time'].dt.hour
    df_unique['publish_day'] = df_unique['publish_time'].dt.day_name()

    # 6. Outlier Removal & Data Filtering
    q_views = df_unique['views'].quantile(0.99)
    df_clean = df_unique[
        (df_unique['days_to_trend'] <= 4) & (df_unique['views'] <= q_views)
        ].copy()

    # 7. Metadata & Formatting
    df_clean['month_trending'] = df_clean['trending_date'].dt.strftime('%Y-%m')
    df_clean['country'] = country_name

    return df_clean


# قاموس الـ 10 دول شاملاً المكسيك (MX)
all_countries_config = {
    'CA': ('CAvideos.csv', 'CA_category_id.json', 'Canada'),
    'US': ('USvideos.csv', 'US_category_id.json', 'USA'),
    'FR': ('FRvideos.csv', 'FR_category_id.json', 'France'),
    'RU': ('RUvideos.csv', 'RU_category_id.json', 'Russia'),
    'IN': ('INvideos.csv', 'IN_category_id.json', 'India'),
    'GB': ('GBvideos.csv', 'GB_category_id.json', 'UK'),
    'DE': ('DEvideos.csv', 'DE_category_id.json', 'Germany'),
    'JP': ('JPvideos.csv', 'JP_category_id.json', 'Japan'),
    'KR': ('KRvideos.csv', 'KR_category_id.json', 'South Korea'),
    'MX': ('MXvideos.csv', 'MX_category_id.json', 'Mexico')  # إضافـة دولة المكسيك
}

all_cleaned_dfs = []

# تشغيل الـ Pipeline أوتوماتيكياً على الـ 10 دول
for code, (csv_path, json_path, country_name) in all_countries_config.items():
    print(f"🔄 Processing dataset for: {country_name}...")
    try:
        df_processed = process_youtube_pipeline(csv_path, json_path, country_name)

        # حفظ CSV منفصل لكل دولة
        df_processed.to_csv(f'{code}_youtube_trending_cleaned.csv', index=False)

        all_cleaned_dfs.append(df_processed)
        print(f"✅ {country_name} processed and exported successfully!")
    except Exception as e:
        print(f"❌ Failed to process {country_name}: {e}")

# دمج الـ 10 دول في ملف Master واحد
df_global = pd.concat(all_cleaned_dfs, ignore_index=True)
df_global.to_csv('youtube_global_10_countries_cleaned.csv', index=False)

print(f"\n🎉 Pipeline Execution Complete!")
print(f"Global Combined Dataset Shape: {df_global.shape}")