import pandas as pd
import os
import mimetypes
import numpy as np
import math
import queue
import matplotlib.pyplot as plt

START_DATE = 1990

class EmissionsData:
    def __init__(self, path=None):
        if path == None:
            try:
                from PyQt5.QtWidgets import QApplication, QFileDialog
            except ImportError:
                raise ImportError("PyQt5 is required for file picking if no file is supplied")
            app = QApplication([])
            path = QFileDialog.getOpenFileName(directory=os.getcwd(),
                                               caption="Please provide a NYDEC csv Emissions report")[0]
            if path == "":
                raise FileNotFoundError("No file was selected")
            app.exit()
        if not os.path.isfile(path) or os.path.islink(path):
            raise TypeError("Path must be a file. Links are not allowed")
        if mimetypes.guess_type(path)[0] != "text/csv":
            raise TypeError("Passed file must be a csv")
        self._dataframe = pd.read_csv(path)
        self.years = self._dataframe['Year'].unique()
        self.gasses = self._dataframe['Gas'].unique()
        self.economic_sectors = self._dataframe["Economic Sector"].unique()
        self.sectors = self._dataframe["Sector"].unique()
        self.categories = self._dataframe["Category"].unique()
        self.first_subcategories = sorted(self._dataframe["Sub-Category 1"].unique())
        self.second_subcategories = sorted(self._dataframe[self._dataframe["Sub-Category 2"] != "Not Applicable"]["Sub-Category 2"].unique())
        self.third_subcategories = sorted(self._dataframe[self._dataframe["Sub-Category 3"] != "Not Applicable"]["Sub-Category 3"].unique())
    
    def categories_by_sector(self, sector):
        newframe = self._dataframe[self._dataframe["Sector"] == sector]["Category"]
        return sorted(newframe.unique())

    def first_subcategories_by_category(self, category, sector=None):
        newframe = self._dataframe[self._dataframe["Category"] == category]
        if sector:
            newframe = newframe[newframe["Sector"] == sector]
        return sorted(newframe["Sub-Category 1"].unique())
    
    def second_subcategories_by_first_subcategory(self, first_subcategory, category=None, sector=None):
        newframe = self._dataframe[self._dataframe["Sub-Category 1"] == first_subcategory]
        if sector: 
            newframe = newframe[newframe["Sector"] == sector]
        if category: 
            newframe = newframe[newframe["Category"] == category]
        return sorted(newframe["Sub-Category 2"].unique())
    
    def third_subcategories_by_second_subcategory(self, second_subcategory, first_subcategory=None, category=None, sector=None):
        newframe = self._dataframe[self._dataframe["Sub-Category 2"] == second_subcategory]
        if sector: 
            newframe = newframe[newframe["Sector"] == sector]
        if category: 
            newframe = newframe[newframe["Category"] == category]
        if first_subcategory: 
            newframe = newframe[newframe["Sub-Category 1"] == first_subcategory]
        return sorted(newframe["Sub-Category 3"].unique())
    
    def get_yearly_emissions(self, metric="MT CO2e AR5 20 yr", years=None, biogenic=None, gas=None,
                        conventional=None, economic_sector=None, 
                        sector=None, category=None, subcategory1=None, 
                        subcategory2=None, subcategory3=None):
        gross_emissions = self.get_yearly_gross_emissions(metric, years, biogenic, gas, conventional, economic_sector,
                                                        sector, category, subcategory1, subcategory2, subcategory3)
        emission_removals = self.get_yearly_removals(metric, years, gas, conventional, economic_sector, sector, category,
                                                     subcategory1, subcategory2, subcategory3)
        return gross_emissions + emission_removals
        
        

    def get_yearly_gross_emissions(self, metric="MT CO2e AR5 20 yr", years=None, biogenic=True, gas=None,
                        conventional=None, economic_sector=None, 
                        sector=None, category=None, subcategory1=None, 
                        subcategory2=None, subcategory3=None):
        newframe = self._dataframe[self._dataframe["Gross"] == "Yes"]
        if biogenic is False:
            newframe = newframe[newframe["Net"] == True]
        return self._calculate_yearly_MTCO2e(newframe, metric, years, gas, conventional,
                                            economic_sector, sector, category, 
                                            subcategory1, subcategory2, subcategory3)
    
    def get_yearly_removals(self, metric="MT CO2e AR5 20 yr", years=None, gas=None, 
                            conventional= None, economic_sector=None,
                            sector=None, category=None, subcategory1=None, 
                            subcategory2=None, subcategory3=None):
        newframe = self._dataframe[(self._dataframe["Net"] == "Yes") & (self._dataframe["Gross"] == "No")]
        return self._calculate_yearly_MTCO2e(newframe, metric, years, gas, conventional,
                                            economic_sector, sector, category,
                                            subcategory1, subcategory2, subcategory3)
    
    def _calculate_yearly_MTCO2e(self, initial_frame, metric="MT CO2e AR5 20 yr", years=None,
                                gas=None, conventional=None, economic_sector=None,
                                sector=None, category=None, subcategory1=None, 
                                subcategory2=None, subcategory3=None):
        if not years:
            years = self.years
        if gas:
            initial_frame = initial_frame[initial_frame["Gas"] == gas]
        if conventional:
            conventional = "Yes" if conventional is True else "No"
            initial_frame = initial_frame[initial_frame["Conventional Accounting"] == conventional]
        if economic_sector:
            initial_frame = initial_frame[initial_frame["Economic Sector"] == economic_sector]
        if sector:
            initial_frame = initial_frame[initial_frame["Sector"] == sector]
        if category:
            initial_frame = initial_frame[initial_frame["Category"] == category]
        if subcategory1:
            initial_frame = initial_frame[initial_frame["Sub-Category 1"] == subcategory1]
        if subcategory2:
            initial_frame = initial_frame[initial_frame["Sub-Category 2"] == subcategory2]
        if subcategory3:
            initial_frame = initial_frame[initial_frame["Sub-Category 3"] == subcategory3]
        yearly_MTCO2e = np.zeros(len(self.years))
        for i in range(len(years)):
            yearly_MTCO2e[i] = initial_frame[initial_frame["Year"] == years[i]][metric].sum()
        return yearly_MTCO2e
    
def binsize(data, method="Rice's"):
    match method:
        case "Sturge's":
            return int(1+(3.322*math.log(len(data), 10)))
        case "Scott's": # (Produces bugs with large standard deviations or quartiles)
            return int(3.49*np.std(data)*(len(data)**(-1/3)))
        case "Freedman-Diaconis's": # (Produces bugs with large standard deviations or quartiles)
            return int((np.percentile(data, 75)-np.percentile(data, 25))*2*(len(data)**(-1/3)))
        case _: # Rice's by default
            return int((len(data)**(1/3))*2)

def generate_granular_graphs(data: EmissionsData, total_type="total",
                            metric="MT CO2e AR5 20 yr", years=None, gas=None, conventional=None):
    if not years:
        years = data.years
    graphs = _granular_graph_helper(data)
    while graphs.qsize() > 0:
        graph = graphs.get()
        _granular_plotter_helper(data, graph, total_type, years, metric, gas, conventional)
        

def _granular_graph_helper(data: EmissionsData):
    graphs = queue.Queue()
    for sector in data.sectors:
        catagories = data.categories_by_sector(sector)
        graphs.put((sector, catagories))
        for category in catagories:
            first_subcategories = data.first_subcategories_by_category(category, sector=sector)
            graphs.put((sector, category, first_subcategories))
            for sub_category_1 in first_subcategories:
                second_subcategories = data.second_subcategories_by_first_subcategory(sub_category_1, category=category, sector=sector)
                if second_subcategories[0] != "Not Applicable":
                    graphs.put((sector, category, sub_category_1, second_subcategories))
                for sub_category_2 in second_subcategories:
                    third_subcategories = data.third_subcategories_by_second_subcategory(sub_category_2, first_subcategory=sub_category_1, category=category, sector=sector)
                    if third_subcategories[0] != "Not Applicable":
                        if sub_category_2 == "Not Applicable":
                            graphs.put((sector, category, sub_category_1, "", third_subcategories))
                            break
                        graphs.put((sector, category, sub_category_1, sub_category_2, third_subcategories))
    return graphs

def _granular_plotter_helper(data: EmissionsData, graph, total_type, years, metric, gas, conventional):
    total_method = None
    title_prefix = None
    match total_type:
        case "gross":
            total_method = data.get_yearly_gross_emissions
            title_prefix = ("Gross ", "emissions")
        case "removals":
            total_method = data.get_yearly_removals
            title_prefix = ("", "removals")
        case _: # total
            total_method = data.get_yearly_emissions
            title_prefix = ("Total ", "emisssions/removals")
    granularity = len(graph)
    emissions = None
    title_str = None
    match granularity:
        case 2:
            title_str = f"{title_prefix[0]}{graph[0]} sector {title_prefix[1]} by category"
            emissions = [total_method(sector=graph[0], category=category, metric=metric, gas=gas, conventional=conventional) for category in graph[1]]
        case 3:
            title_str = f"{title_prefix[0]}{graph[0]} sector, {graph[1]} {title_prefix[1]} by Sub-Category 1"
            emissions = [total_method(sector=graph[0], category=graph[1], subcategory1=subcategory1, metric=metric, gas=gas, conventional=conventional) for subcategory1 in graph[2]]
        case 4:
            title_str = f"{title_prefix[0]}{graph[0]} sector, {graph[1]}, {graph[2]} {title_prefix[1]} by Sub-Category 2"
            emissions = [total_method(sector=graph[0], category=graph[1], subcategory1=graph[2], subcategory2=subcategory2, metric=metric, gas=gas, conventional=conventional) for subcategory2 in graph[3]]
        case 5:
            sub_category_2 = None
            if graph[4] == "":
                title_str = f"{title_prefix[0]}{graph[0]} sector, {graph[1]}, {graph[2]} {title_prefix[1]} by Sub-Category 3"
                sub_category_2 = "Not Applicable"
            else:
                title_str = f"{title_prefix[0]}{graph[0]} sector, {graph[1]}, {graph[2]}, {graph[3]} {title_prefix[1]} by Sub-Category 3"
                sub_category_2 = graph[3]
            emissions = [total_method(sector=graph[0], category=graph[1], subcategory1=graph[2], subcategory2=sub_category_2, subcategory3=subcategory3, metric=metric, gas=gas, conventional=conventional) for subcategory3 in graph[4]]
    emissions = np.array(emissions)
    if np.all(emissions == 0): return
    if title_prefix[1] == "removals": emissions = abs(emissions)
    plt.title(title_str)
    plt.ylabel("Years")
    plt.ylabel(f"{metric} ({title_prefix[1]})")
    emission_cnt = -1
    for lowest_level in graph[granularity - 1]:
        emission_cnt += 1
        if np.all(emissions[emission_cnt] == 0): continue
        plt.plot(years, emissions[emission_cnt], label=lowest_level)
    plt.legend()
    plt.show()