
import pandas as pd
from scipy.spatial.distance import cosine

movies= pd.read_csv("C:/Users/xumic/Desktop/Grad scholl/sem3 spring 2021/625/project/data/small/ml-latest-small/movies.csv")
ratings=pd.read_csv("C:/Users/xumic/Desktop/Grad scholl/sem3 spring 2021/625/project/data/small/ml-latest-small/ratings.csv")


# list of possible genres
l=[' '.join(movies['genres'])]
genre_list=""+l[0]
genre_list=genre_list.replace('|',' ')
genre_list=set(genre_list.split(' '))
genre_list.remove('(no')
genre_list.remove('genres')
genre_list.remove('listed)')
genre_list=sorted(genre_list)
print("List of possible genres: ")
print(genre_list)

#variable for preference for different genres by one user
genre_score=pd.DataFrame(columns=genre_list)
genre_score.loc[0]=0


#function to convert list into string
def list_to_string(s): 
    str1 = "" 
    for ele in s: 
        str1 += str(ele)
    return str1



# see the average ratings for each movies

average_rating1=ratings.groupby('movieId')['rating'].mean().reset_index()\
						.rename(columns={'rating':'average rating'})
average_rating=average_rating1.sort_values(by='average rating',ascending=False)
print("average ratins: ")
print(average_rating)



# construct user profile for the target user, currently set on user 1
user_profile1=ratings[ratings['userId']==1]
user_profile1=pd.merge(user_profile1, movies, on='movieId')

# genre score: preference scores of different genres for user1, caculated by adding up user1's past ratings, will be used as weight
i=0
for x in user_profile1['genres']:
	for y in set(x.split('|')):
		genre_score[y][0]=genre_score[y][0]+user_profile1['rating'][i]
	i+=1
print(genre_score)

# produce genre list and genre score list, the order is important, so the genre and it's score match up
test_data=movies
for z in user_profile1['movieId']:
	test_data=test_data[test_data.movieId!=z]
print(test_data)
genre_score_list=genre_score.values.tolist()
genre_score_list=genre_score_list[0]

genre_score_list, genre_list=zip(*sorted(zip(genre_score_list,genre_list),reverse=True))
genre_score_list, genre_list=(list(q) for q in zip(*sorted(zip(genre_score_list,genre_list),reverse=True)))
#print(genre_list)
#print(genre_score_list)
#print(len(genre_list))




# modified movies data with genre vector: which is represented by a number string, with 1 meaning the movie belonging to the genre, the order is the same as the genre list
print("movie data set with genre vector: ")
movies_G_vector=movies
movies_G_vector["genre_vector"]="0000000000000000000"

row_count=0
for n in movies_G_vector["genre_vector"]:
	index_count=-1
	temp_list=[int(s) for s in n] #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	for genre in genre_list:
		index_count+=1
		if genre in movies_G_vector["genres"][row_count].split("|"):
			temp_list[index_count]+=1
	movies_G_vector["genre_vector"][row_count]=list_to_string(temp_list)
	row_count+=1
print(movies_G_vector)

# generate score using cosine similarity between the genre vector for the movie and the comparison vector in which all of the values are 1, with the genre score list as the weight factor, this can generate good recommendation because of the way the scipy cosine similarity works, also, the system does not take negative feedback into account, and the genres are weighted
comparison_vector=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
print("movie data set with scores")
movies_G_vector["prediction_score"]=0.0
genre_weight_list=genre_score_list

row_count2=0
for n in movies_G_vector["genre_vector"]:
	movies_G_vector["prediction_score"][row_count2]=cosine(comparison_vector,[int(s) for s in n],genre_weight_list)
	row_count2+=1

print(movies_G_vector)
#The lower the score the higher the movie should be on the recommended list
final_list=movies_G_vector.sort_values(by=["prediction_score"])
print("final result")
print(final_list)
