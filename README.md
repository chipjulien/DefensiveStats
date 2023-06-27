# DefensiveStats

Statical Metrics for Defense in Basketball

#Introduction

This project is a prototype concept of a vision system that will accurately quantify defensive metrics in basketball.
This project will scale to analyzing live feed video but, currently, we parse through static mp4 files. Future plans to add advanced 
defensive statistics.

#Background

This project was created to fill the gap in statistical metrics captured while playing defense in basketball. These metrics are 
commonly known as the intangible attributes of a player. During the regular season, you can view players such as Marcus Smart, Tony 
Allen, and Rudy Gobert night after night performing to the highest caliber but, the stat sheet depicts a subpar trial. The reason for 
this is because "it's a make or miss league" a quote Ouzing offensive bias. along with the intangibles, any discrepancies in the 
current approach of collecting each datum will be reviewed and reanalyzed. Ie if a player blocks a two-point shoot they get 1 block 
but, if they block a three-point shoot they receive the same amount of blocks as if they blocked a two-point shoot. The vision system 
will track defensive assist which is a new concept that will allow for more granularity in each of the datums. Defensive assists are 
metrics to award players engaged in a rebound or steal that would've been awarded to other players not fully engaged in the defensive 
play. More features and metrics will be added as the project grows.

#Objectives

This project will build an engine that will parse visual data in the form of basketball games and deliver a score based on the 
defensive metric algorithms created to analyze visual defensive intangibles. The first module will track the ball in an mp4 file and 
clip the video into subdivisions of plays segmented by the number of times the ball crosses the halfcourt line. These subdivisions 
will be parsed through the defensive algorithm and scored. These scores will be for an individual player which will be collected and 
scored together for team results.

#balltracking.py

Uses opencv, numpy, and moviepy libraries to clip subdivisions of video from the larger mp4 video.
We track the ball with the Shi-Tomasi corner detection method, then decide which side of the halfcourt the ball is currently on.
Each time ball_location != prev_ball_location the file will be segmented and saved as play 1,2,3...

#TODO

-Debug balltracking.py - May need to move the project into C++

-Build defensive Algo's
