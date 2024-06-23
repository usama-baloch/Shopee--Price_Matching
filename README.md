# Shopee--Price_Matching

## Description:

Do you scan online retailers in search of the best deals? You're joined by the many savvy shoppers who don't like paying extra for the same product depending on where they shop. Retail companies use various methods to assure customers that their products are the cheapest. Among them is product matching, which allows a company to offer products at competitive rates to the same product sold by another retailer. To perform these matches automatically requires a thorough machine learning approach, which is where your data science skills could help.

Two different images of similar wares may represent the same product or two completely different items. Retailers want to avoid misrepresentations and other issues from conflating two dissimilar products. Currently, a combination of deep learning and traditional machine learning analyzes image and text information to compare similarities. However major differences in images, titles, and product descriptions prevent these methods from being entirely effective.

[Shopee](https://shopee.com/) is the leading e-commerce platform in Southeast Asia and Taiwan. Customers appreciate its easy, secure, and fast online shopping experience tailored to their region. The company also provides strong payment and logistical support along with a 'Lowest Price Guaranteed' feature on thousands of Shopee's listed products. 

## What do we need to do about this?

we will overcome this problem by using our Machine Learning and/or Deep Learning skills to build a pipeline that predicts which items are for the same product.

### Note:

It was a Kaggle Competition and it is a real-world problem to tackle, you can design your solution by helping multiple retailers, and your solution to product matching could support more accurate product categorization and reduce misrepresentation. 

You can find the dataset from here: [Shopee Dataset](https://www.kaggle.com/competitions/shopee-product-matching/data)

Evaluation Metric used F1 Score.

## Solution:

1) we need to find the similarities between products by either using images or texts or both of them, I tried using the images and texts to find the similarities.

2) Designed a Model 'eca_nfnet_l0' and made small changes in that which now returns the image embeddings instead of returning a one-hot encoded vector of classes.
Explanation:
In this problem, we need to match the products by their images and the best way to do that is to find the similarities of one image embedding with another. so the question, is
what is embedding? Embedding in the sense of Computer Vision are the features extracted from the feature extractor of eca_nfnet_l0. but we learn that the CNN Classifiers only return the probabilities of the classes?
Here we are extracting the embeddings before the last layer which is the output layer which return the probabilities of the classes.
So Embedding is nothing but numbers which are useful because they are came from the number of Layers of CNN Classifier and they have important data like the uniqueness of the image. 

3) Now we get the embeddings of the images using step 2 but how we gonna confirm that this image is similar to the comparing image?
Here: we find the distance between these two embeddings using the cosine distance formula, which is one minus the cosine of the angle from point one to the origin to point two. when this is equal to 0 it means these two images are the same and if it is 1 it means these are different.
ArcFace: We would like similar classes to have embeddings close to each other and dissimilar ones far from each other, This is what ArcFace does. ArcFace adds more loss to the training procedure to encourage similar class embeddings to be close and dissimilar classes to be far from each other.

4) Now After finishing steps 2 and step 3 we find the text embeddings using different techniques, I used TFIDF the concept is the same which is to find similarities, In the text we try to find similar titles.
5) When we get the image predictions and text predictions then we just combine those and get the results.
6) The results will be like this: Product A: A B C G, this means product A is similar to the products A, B, C, and G.

Extra Info: 

[Great Explanation about ArcFace and Embeddings](https://www.kaggle.com/c/shopee-product-matching/discussion/226279)

[ArcFace Implementations](https://www.kaggle.com/code/slawekbiel/arcface-explained)
