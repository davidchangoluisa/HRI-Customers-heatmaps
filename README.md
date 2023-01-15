# Heatmaps for tracking customer behaviou

This ROS package contains the implementations of a  

---

### About this project

<b>Autors</b>:

Ivan, Changoluisa
Ahmad, Alrashed <br> 

This project has been created as part of the Human-Robot Interaction course of the IFROS Master in the University of Zagreb.

---
<!-- ### Code description 

**People Heatmap generator** 



**Customer Heatmap generator**
 -->


## User's Guide 
---
To test the system, some input videos are included in the folder /test_videos. 

**Testing the People Heatmap**

The first argument is the path of the input video and the second is the desired name for the resulting video that will be generated after the test. 

```console
    python ./people_heatmap.py test_videos/urban.mp4 urban_people_heatmap
```
 After this, a result similar as the one below will be shown.
 
 <img src="results/customer_urban.gif" width="800">
 


**Testing the Customer Heatmap**
 Similarly as for the People Heatmap, the 

```console
    python ./customer_heatmap.py test_videos/urban.mp4 urban_customers_heatmap
```
 After this, a result similar as the one below will be shown. 

 <img src="results/people_urban.gif" width="800">


<!-- 
## Future work and To-do list
- [ ] 
 
 -->
<!-- 
## Conclusions
---
- An occupancy grid map generator has been successfully implemented in this ROS package, the generator makes use of laser scans and is able to determine whether a cell is free or occupied by following a probabilistic approach. 
- The popular open-source occupancy map generator called Octomap Server was also tested and included in this package. Also, the guidelines to properly set up this package has been detailed. 
  


```python

``` -->

