# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 09:22:09 2020

@author: Hitesh
"""

import glassdoor_scraping_using_selenium as gs
import pandas as pd 

data=gs.get_jobs('data scientist', 956╝, False)
data.to_csv("glassdoor.csv",index=False)


