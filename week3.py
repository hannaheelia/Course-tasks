import pandas as pd

# Loading the dataset
df = pd.read_csv('Airbnb_ratings_NYC_2019.csv')

# Finding the median price of the Manhattan hotels
median_price_manhattan_hotels = df[
    (df['neighbourhood_group'] == 'Manhattan')&
    (df['host_name'].str.lower().str.contains('hotel')) # assumption that host name include hotel if it is a hotel
    ]['price'].median()

#Number of entire homes/apt in SoHo
count_apt__soho = df[
    (df['neighbourhood']=='SoHo')&
    (df['room_type']== 'Entire home/apt')
    ].shape[0]

#Number of hosts in Williamsburg with to or more entire homes/apt
apt_no_williamsburg = df[
    (df['neighbourhood']=='Williamsburg')&
    (df['room_type']=='Entire home/apt')
    ]

apt_count_williamsburg = apt_no_williamsburg.groupby('host_id').size() # Grouping by host_id
over_two_host_w = (apt_count_williamsburg>2).sum() # get hosts with two or more houses/apartments and sum them

# Answers
print("a) Median price for hotel listings in Manhattan:", median_price_manhattan_hotels)
print("b) Number of entire homes/apartments in SoHo:", count_apt__soho)
print("c) Number of hosts in Williamsburg with to or more entire homes/apt:", over_two_host_w)

# TASK 2

def reviewFilter():
    neighbourhood = input("Neighbourhood to filter by: ")
    df_by_neighbourhood = df[(df['neighbourhood']== neighbourhood)&
                             (df['host_name'].str.lower().str.contains('hotel'))]
    if df_by_neighbourhood.empty:
        print("No matching results, check neighbourhood name")

    minimum_price = int(input("Minimun price to filter by: "))
    df_by_minimum_price = df_by_neighbourhood[df_by_neighbourhood['price']>=minimum_price]
    if df_by_minimum_price.empty:
        print("No matching results")
    
    minimum_number_of_reviews = int(input("Minimum number of reviews to filter by: "))
    df_by_review_no = df_by_minimum_price[df_by_minimum_price['number_of_reviews']>=minimum_number_of_reviews]
    if df_by_review_no.empty:
        print("No matching results")
    else:
        filteredReviews = df_by_review_no.loc[:,['host_name','neighbourhood','price','number_of_reviews']]
        filteredReviews.sort_values(by='number_of_reviews', ascending=[False])
        print(filteredReviews.head(10))
        return filteredReviews

filtered_reviews = reviewFilter()
hotels_in_filtered = filtered_reviews.groupby('host_name')
hotel_count_filtered = hotels_in_filtered.ngroups
hotel_names_filtered = hotels_in_filtered.size().index
host_count_filtered = df[df['host_name'].isin(hotel_names_filtered)].groupby('host_id').ngroups

print('Number of hotels in filtered results:', hotel_count_filtered)
print('Number of host in filtered results:', host_count_filtered)
print()

# TASK 3
summary_filter = df[
    (df['price']>=50)&
    (df['number_of_reviews']>=5)&
    ((df['host_name'].str.lower().str.contains('hotel')))
     ]
neighbourhoods = summary_filter.groupby('neighbourhood').size().index
print(neighbourhoods)
for neighbourhood in neighbourhoods:
    df_neighbourhood = summary_filter[summary_filter['neighbourhood']==neighbourhood]
    hotels = df_neighbourhood.groupby('host_name')
    hotel_count = hotels.ngroups
    avg_price = df_neighbourhood['price'].mean()
    print('Neighbourhood:', neighbourhood)
    print('Number of hotels listed:', hotel_count)
    print('Average price of hotels:', avg_price)
    print()
