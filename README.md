# ENGR-1330-Final-Project
The following is the final project of my ENGR-1330 class at Texas Tech University. The aim of the project is to provide tools for the analysis of the NYSDEC Greenhouse Gas Emissions Report. The report publishes the annual metric tons of carbon dioxide equivalent (with 20 or 100 year GWP adjustment) as per the Intergovernmental Panel on Climate Change's Fith and Fourth Annual Reports respectively scince 1990. The data is broken up into a variety of economic sectors, subsectors, gas types, and reporting types. The aim of the Greenhouse Gas Emissions Report is to track the progress towards the Climate Leadership and Community Protection Acts (CLCPA) requirements. Of those requirements are a reductions of the MT CO2e AR5 20 yr (metric tons of carbon dioxide equivalent 20 year GWP adjustment as per IPCC 5th annual report) to 60% of 1990 levels by 2030 and 15% of 1990 levels by 2050.

## Distributions, Plots and Descriptive Statistics
![Figure_1](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/aa106dc0-3c73-4a80-b2c9-99384e560fc2)
![Screenshot from 2024-05-04 17-48-25](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/4409147a-f61a-41ad-8daa-9b648ac8fe4f)
![Figure_2](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/30f16d13-6501-4a61-99f2-e1b32fc8564c)
![image](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/908eeb19-7bbc-43a0-8298-718c25e48211)
![Figure_3](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/b3f88645-b679-4316-a74f-205dc416db5c)
![Screenshot from 2024-05-04 17-50-53](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/fb576710-765c-45b6-bbd7-fbc4c97bb155)
![Figure_4](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/458f27fb-f818-4888-a299-18a86efb0e38)
![image](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/b34705a0-1476-4716-9dd0-0aa11ebae939)
### Gas Breakdown
![Figure_5](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/588022d1-e02c-493b-9078-904a657a2583)
![Figure_6](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/3ec10509-7d8a-4eed-9d38-fd79f9c46c49)
![Figure_7](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/fe1e66e0-8510-4c9a-8133-72c6011df611)
![Figure_8](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/72d67c87-9335-45df-ad14-537567ca6351)

### Economic Sector Breakdown
For the following graphs, I will only be displaying the MT CO2e AR5 20 yr metric. Using the encapsulating class, it is trivial to query for MT CO2e AR4 100 yr metric.
![Figure_9](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/51257c48-b264-4293-a71b-8c3d2dca3401)
### Sector Breakdown
![Figure_10](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/b92b0bea-0f4f-482c-9401-317cc6c45055)
#### Category Breakdown
![Figure_11](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/57c9c04f-bac0-4c55-ae3f-2f2e312902bc)
### Removals (Sub-category-3) Breakdown
Sub-category-3 is the most granular subcategory
![Figure_12](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/7ca3b709-dc40-438e-927f-00d3e3833e00)
### Arbitrary Queries
Of course, the wrapper can handle many kinds of useful queries, such as...
![Figure_32](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/01623c3a-fb0d-4904-aaa8-2c81d6c5e7f4)
![Figure_33](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/2686c135-7f6f-47ea-af54-70ed6f073ecd)
![Figure_34](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/51846303-4eae-41aa-ac14-0f40084a3295)
![Figure_35](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/c0c246df-80f8-4b90-8b32-c5cf6e02091b)
![Figure_36](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/302139cc-53da-4de3-977d-eacbacdc282b)

## Extrapolation
I would like to extrapolate the data to make simple predictions regarding the progress towards the CLCPA targets.

Here is the MT CO2e AR5 20 yr gross in 1990
![Screenshot from 2024-05-04 18-08-33](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/e8e696a6-7d53-42ad-8f6f-b2e497d56db0)
### Simple Linear Model
- Model
![Figure_13](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/a0ef5dd0-8ce2-4877-8257-89ab1c4394e5)
- Prediction
![Figure_14](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/92c5cee1-a9ab-4079-af39-139e349eb397)
![Screenshot from 2024-05-04 22-37-49](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/29283fa3-dbeb-4e59-8ae5-8623d6c025a2)
- Standardized Residuals
![Figure_15](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/5ae2ead6-598a-40df-9a20-b8428955c1cb)
### Second Order Polynomial Model
- Model
![Figure_20](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/f143ae72-dead-4c17-b3a1-639a4d8c762d)
- Prediction
![Figure_21](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/019470fa-4825-411d-867e-8fc571be2d53)
![Screenshot from 2024-05-04 22-38-33](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/a00e303e-666f-4b4e-a41d-4497c86e0fc6)
- Standardized Residuals
![Figure_22](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/c0e14773-ba78-4144-bef9-348ffcd92853)
![Figure_23](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/e49923b2-a3c1-474c-bd40-3f5db1caae13)
### Third Order Polynomial Model
- Model
![Figure_16](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/86836ba3-83d5-4e04-a6eb-6173c40707e7)
- Prediction
![Figure_17](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/16e8dde8-d591-42a8-bbe7-27544f20f9ba)
![Screenshot from 2024-05-04 22-39-08](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/873e754d-e440-4fa3-9650-f1e47048281b)
(Why you shouldn't trust models when extrapolating)
- Standardized Residuals
![Figure_18](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/0e25636f-b512-4313-90f1-f24e4169539f)
![Figure_19](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/ddc58899-2ae0-46f1-8d42-fe5c72bd021a)
### SVR Model
- Model
![Figure_24](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/999c314c-be3c-4148-8c85-418bdfc6f47b)
- Prediction
![Figure_25](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/81b02a63-ca00-48b4-8884-44387999a83f)
![Screenshot from 2024-05-04 22-39-47](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/5f6bf3e3-491d-41ff-ae50-8002ef946ae5)
- Standardized Residuals
![Figure_26](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/49974430-40e1-412c-be96-14c9d917a5b2)
![Figure_27](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/723d9775-bd78-4f67-9e1f-7810e6de3a99)
### Extrapolation Analysis
Here we train all models again but with a 75/15/15 extrapolation split to curb variance and to see which model might extrapolate the best
#### Linear Model
![Figure_28](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/4e68b01a-533b-4ec6-96e5-2d5ca662b5e6)
![image](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/a44f8e66-8bcb-40bb-91ee-1084c0f91beb)
#### Second Order Polynomial Model
![Figure_30](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/8c3360f7-3ffc-48b6-afaf-a08bc2cf9dd8)
![image](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/b968fe84-6c01-4a11-8d1d-131bd98cea7c)
#### Third Order Polynomial Model
![Figure_29](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/0d5d4561-7e30-4d82-8980-01c1633ee3fc)
![Screenshot from 2024-05-04 18-33-56](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/b3090b91-f2f1-4ac7-a070-a1e19aa89ec4)
#### SVR Model
![Figure_31](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/52dbc33c-1073-4050-a804-edfafb839d12)
![Screenshot from 2024-05-04 18-37-44](https://github.com/Colby-Coffman/ENGR-1330-Final-Project/assets/114829458/80a63e01-3a53-4365-997a-9a6f2d5c36f4)
Funnily enough the linear model performed best at extrapolation.
