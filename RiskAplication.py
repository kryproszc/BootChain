import pandas as pd
import numpy as np
from operator import itemgetter 
import calendar
import sys
import tkinter as tk
from datetime import date, datetime
from tkinter import ttk
from tkinter.font import nametofont
from tkinter.filedialog import askopenfile
import openpyxl as oxl
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandastable import Table
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from operator import itemgetter 
from scipy.stats import gamma


class BootStrapChian:
    def __init__(self, df_t, R):
        self.df_t = df_t
        self.R = R
        self.IBNR_all = pd.DataFrame()


    def Union(self,lst1, lst2):
        final_list = list(set(lst1) | set(lst2))
        return final_list
        
    def delete_element_list(self,list_1,list_2):
        for item in list_2:
            if item in list_1:
                list_1.remove(item)
        return(list_1)

    def index_all(self,df_t):
        df_trian = self.df_t
        index_he = []
        index_ost = []
        index_firs = []
        m = df_trian.shape[0]
        n = df_trian.shape[1]
        for i in range(n-1):
            ind_row = [x for x in range(m)]
            ind_col_last = np.where(df_trian.iloc[:,i+1].isnull())[0].tolist()
            ind_col_before = np.where(df_trian.iloc[:,i].isnull())[0].tolist()
            index_ost.append(self.delete_element_list(ind_row,ind_col_before).pop())
            index_firs.append(min(self.delete_element_list(ind_row,ind_col_last)))
            sum_ind = self.Union(ind_col_last,ind_col_before)
            dev_ind = self.delete_element_list(ind_row,sum_ind)
            index_he.append(dev_ind)
        index_ost.append(ind_col_last[0]-1)
        return(index_he,index_ost,index_firs)

    def wyznacz_f_1(self,df_triangle):
        indeksy, _ ,_= self.index_all(df_triangle)
        f = []
        n = df_triangle.shape[1]
        for i in range(n-1):
            f.append(sum(df_triangle.iloc[indeksy[i],i+1].dropna().astype(float))/sum(df_triangle.iloc[indeksy[i],i].dropna().astype(float)))
        return(f)    

    def trian_diag(self,df_triangle):
        n = df_triangle.shape[1]
        _, ind_ost,_ = self.index_all(df_triangle)
        el = [df_triangle.iloc[i,j] for i,j in zip(ind_ost,range(n))]
        return(el)

    def reverse_list(self,arr):
        left = 0
        right = len(arr)-1
        while (left < right):
            # Swap
            temp = arr[left]
            arr[left] = arr[right]
            arr[right] = temp
            left += 1
            right -= 1
        return(arr)
    
    def iloczn_wstepujacy(self,lista):
        lista_new = []
        lista_new.append(lista[len(lista)-1])
        for i in range(len(lista)-1):
            lista_new.append(lista_new[i]*lista[len(lista)-2-i])
        return(lista_new)
    
    def add_el_list(self,a,b):
        pp = []
        for i in range(a,b+1,1):
            pp.append(i)
        return(pp)
    
    def row_list_all(self,list1,list2,nn):
        krotka = list(zip(list1,list2))
        row_list = []
        for i in range(nn-1):
            kr = krotka[nn-i-2]
            row_list.append(self.add_el_list(kr[0],kr[1]))
        return(row_list)
    
    def triangle_back(self,df_t,l1,l2,ults,f_p):
        n = df_t.shape[1]
        df_t_copy = df_t.copy()
        row_list = self.row_list_all(l1,l2,n)
        for j in range(n-1):
            y = f_p[j]
            df_t_copy.iloc[row_list[j],n-j-2] = [x/y for x in list(itemgetter(*row_list[j])(ults))] 
        return(df_t_copy)
    
    def incremental_triangle(self,df_triangle):
        indeksy, _ ,_= self.index_all(df_triangle)
        n = df_triangle.shape[1]
        df_trian_copy = df_triangle.copy()
        for i in range(n-1):
            b = df_triangle.iloc[indeksy[i],i+1] - df_triangle.iloc[indeksy[i],i]
            df_trian_copy.iloc[indeksy[i],i+1] = b
        return(df_trian_copy)
    
    def triangle_back_down_triangle2(self,df_t,l2,ult,f_p):
        ult_cop = ult
        m = df_t.shape[0]-1
        n = df_t.shape[1]
        f_p_rev = (f_p)
        df_t_copy = df_t.copy()
        df_t_copy[:] = np.nan
        len_org = len(df_t.iloc[:,n-1])
        len_ost = len(ult)
        r = len_org-len_ost
        ult_copy = ult
        r_copy = r
        if r>0:
            for i in range(r):
                ult_copy.insert(i,0)
        df_t_copy.iloc[:,n-1] = ult_copy
        for i in range(n-r_copy):
            ind_row_i = [a for a in range(r,m+1,1)]
            y = f_p_rev[i]

           # print(ind_row_i)
            if len(ind_row_i) >1:
                l_pom = [x/y for x in list(itemgetter(*ind_row_i)(ult_copy))]
                for j in range(r):
                    l_pom.insert(j,0)


                df_t_copy.iloc[:,n-i-2] = l_pom
            else:
                l_pom = [ult_copy[ind_row_i[0]]/y]
                for i in range(r):
                    l_pom.insert(i,0)
                df_t_copy.iloc[:,n-i-2] = l_pom
            r = r+1
        return(df_t_copy)

    def rev_incremental(self,dd):
        dd_copy = dd.copy()
        n = dd_copy.shape[1]
        dd_0 = dd_copy.fillna(0)  
        dd_copy_0 = dd_copy.fillna(0)  
        for i in range(n-1):
            dd_copy_0.iloc[:,i+1] = dd_0.iloc[:,i+1] - dd_0.iloc[:,i]
        return(dd_copy_0)
    
    def elements_triangle(self,df_dd,ult2):
        out_list = []
        ul_cop2 = ult2
        m = df_dd.shape[0]-1
        n = df_dd.shape[1]
        len_org = len(df_dd.iloc[:,n-1])
        len_ost = len(ult2)
        r = len_org-len_ost
        ind_row_i = []
        r_copy = r
        for i in range(n-r_copy):
            ind_row_i.append([a for a in range(r,m+1,1)])
            r = r+1
        for i in range(n-r_copy):
            df_dd.iloc[:,i+r_copy]
            for x in df_dd.iloc[ind_row_i[n-r_copy-i-1],i+r_copy].tolist():
                out_list.append(x)
        return(out_list)
    
    def add_gamma_to_pdf(self,df_dd,ult2,w_gamma):
        znak_triangle = np.sign(df_dd)
        process_triangl = pd.DataFrame(0, columns=znak_triangle.columns, index=znak_triangle.index)
        out_list = []
        ul_cop2 = ult2
        m = df_dd.shape[0]-1
        n = df_dd.shape[1]
        len_org = len(df_dd.iloc[:,n-1])
        len_ost = len(ult2)
        r = len_org-len_ost
        r_copy = r
        k = 0
        for i in range(n-r_copy):
            ind_row_indy = [a for a in range(m-r+1,m+1,1)]
            r = r+1
            for j in ind_row_indy:
                if(znak_triangle.iloc[j,i+r_copy]==1.0):
                    process_triangl.iloc[j,i+r_copy] = w_gamma[k]
                elif(znak_triangle.iloc[j,i+r_copy]==-1.0):
                    process_triangl.iloc[j,i+r_copy] = -w_gamma[k]
                k = k+1
        return(process_triangl)

    def Boot_strap(self):
        m = self.df_t.shape[0]
        n = self.df_t.shape[1]
        f = self.wyznacz_f_1(self.df_t)
        f_prod = self.iloczn_wstepujacy(f)
        diagonala = self.trian_diag(self.df_t)
        diagonala = diagonala[0:(len(f_prod))]
        diagonala_rev = self.reverse_list(diagonala)
        n = self.df_t.shape[1]
        ults = [diagonala_rev[i]*f_prod[j] for i,j in zip(range(n-1),range(n-1))]
        len_org = len(self.df_t.iloc[:,n-1])
        len_est = len(ults)
        r = len_org-len_est
        if r>0:
            for i in range(r):
                ults.insert(i,self.df_t.iloc[i,n-1])
        _,ind_dol,ind_gora = self.index_all(self.df_t)
        l1,l2 = ind_gora,ind_dol[:-1]        
        back_tr = self.triangle_back(self.df_t,l1,l2,ults,f_prod)
        exp_inc_triangle = self.incremental_triangle(back_tr)
        inc_triangle = self.incremental_triangle(self.df_t) 
        nobs = 0.5*(self.df_t.shape[1])*(self.df_t.shape[1]+1)
        scale_factor = (nobs - 2*self.df_t.shape[1]+1)
        res_triangle = (inc_triangle - exp_inc_triangle)/np.sqrt(np.abs(exp_inc_triangle))
        res_triangle = round(res_triangle,5)
        adj_res = res_triangle*round(np.sqrt(nobs/scale_factor),6)
        scale_phi = (np.sum(res_triangle**2)/scale_factor).sum() 
        df = pd.DataFrame(
                    columns =list(self.df_t.index))
        df["Suma"] = []
        for iteracje in range(self.R):
                diagonala_rev = ''
                diagonala_p = ''
                choices = adj_res.values[~pd.isnull(adj_res.values)]
                losowa_macierz = adj_res.applymap(lambda x: np.random.choice(choices) if not pd.isnull(x) else x)
                new_triangle = losowa_macierz*np.sqrt(np.abs(exp_inc_triangle))+exp_inc_triangle
                suma_kum = new_triangle.cumsum(axis = 1)
                f_przejscia = self.wyznacz_f_1(suma_kum)
                f_prod = self.iloczn_wstepujacy(f_przejscia)
                diagonala = self.trian_diag(suma_kum)
                diagonala_p = diagonala[0:(len(f_prod))]
                diagonala_rev = self.reverse_list(diagonala_p)
                n = self.df_t.shape[1]
                ults_loop = [diagonala_rev[i]*f_prod[j] for i,j in zip(range(n-1),range(n-1))]
                tr_down = self.triangle_back_down_triangle2(self.df_t,l2,ults_loop,f_prod)
                ults_loop_2 = [diagonala_rev[i]*f_prod[j] for i,j in zip(range(n-1),range(n-1))]
                rev_triangle = self.rev_incremental(tr_down)
                list_element = self.elements_triangle(rev_triangle,ults_loop_2)
                scale = [np.abs(x/scale_phi) for x in list_element]
                wartosci_gamma = gamma.rvs(scale, scale=scale_phi, size=len(scale))
                processTriangle = self.add_gamma_to_pdf(rev_triangle,ults_loop_2,wartosci_gamma)
                IBNR = processTriangle.cumsum(axis = 1)
                df.loc[len(df.index)] = IBNR.iloc[:,n-1].to_list() + [np.sum(IBNR.iloc[:,n-1].to_list())]
        IBNR_all = df.copy()
        print(IBNR_all)

        return(IBNR_all)


class AppChainladder(tk.Frame):
    products = []
    #df_app_trian= pd.DataFrame()
    df_app_trian = pd.DataFrame()
    df_app_trian_BOOT = pd.DataFrame()
    
    def __init__(self, root):
        super().__init__(root)
        self.create_UI(root)
    
    def create_UI(self, root):
        if sys.platform.startswith("linux"):
            root.attributes("-zoomed", True)
        else:
            root.state("zoomed")

        self.left_frame  =  tk.Frame(root,  width=500,  height=  750)
        self.left_frame.grid(row=0,  column=0, sticky=tk.NW)

        self.right_frame_top  =  tk.Frame(root,  width=950,  height=375)
        self.right_frame_top.grid(row=0,  column=1, columnspan =2,sticky=tk.NW)

        self.right_frame_title_left = tk.Frame(root,  width=500,  height=30)
        self.right_frame_title_left.grid(row=1,  column=1, sticky=tk.NW)

        self.rihht_frame_title_right = tk.Frame(root,  width=150,  height=30)
        self.rihht_frame_title_right.grid(row=1,  column=2, sticky=tk.NW)

        self.right_frame_chart_left  =  tk.Frame(root,  width=500,  height=375)
        self.right_frame_chart_left.grid(row=2,  column=1, sticky=tk.NW)

        self.right_frame_chart_right  =  tk.Frame(root,  width=150,  height=375)
        self.right_frame_chart_right.grid(row=2,  column=2, sticky=tk.NW)
    
        self.right_chart_title_label = tk.Label(self.rihht_frame_title_right, text="")
        self.right_chart_title_label.pack()

        self.left_chart_title_label = tk.Label(self.right_frame_title_left, text="")
        self.left_chart_title_label.pack()
        
        self.load_data(root)
        self.BootStrap(root)
        self.show_product_combobox(root)
        self.show_all_BOOT(root)

    def open_file(self):
        self.file = askopenfile(mode ='r', filetypes =[('Excel Files', '*.xlsx *.xlsm *.sxc *.ods *.csv *.tsv')]) # To open the file that you want. 
        wb = oxl.load_workbook(filename = self.file.name) 
        app_trian = pd.DataFrame(wb.active.values)
        self.df_app_trian = app_trian.iloc[1:,1:]
        self.products = [str(int(a)) for a in app_trian.iloc[1:,0].tolist()] + ["Suma"]
        table = Table(self.right_frame_top, dataframe=app_trian.iloc[1:,], width=750, height=280)
        table.grid(row=0,column = 0)
        table.show()
        ind_row = [x+1 for x in range(len(self.products)-1)]
        ind_row = ind_row + ["Suma"]
        self.DICTio = dict(map(lambda i,j : (i,j) , self.products,ind_row))

    def load_data(self,root):
        l = tk.Label(self.left_frame,text = "Wprowadź trójkąt z danymi")
        l.grid(row=0,column = 0 , pady = 10,sticky=tk.W)
        btn = tk.Button(self.left_frame, text ='Wczytaj trójkąt', command = self.open_file)
        btn.grid(row=1,column = 0,columnspan = 2)

    def show_chart_btn_clicked(self):
        print(self.product_sv.get())
        
    def show_chart_button(self, root):
        self.show_chart_btn = ttk.Button(
            root, text="TESTY", command=self.show_chart_btn_clicked)
        self.show_chart_btn.grid(row=7, column=1, padx=1)

    def BOOTChain(self):
        boot_with_class = BootStrapChian(self.df_app_trian, 999)
        self.df_app_trian_BOOT = boot_with_class.Boot_strap()
    
    def BootStrap(self,root):
        l = tk.Label(self.left_frame,text = "Uruchoma Bootstrap dla Chainladder")
        l.grid(row=2,column = 0  , pady = 10,sticky=tk.W)
        self.show_chart_btn = ttk.Button(
            self.left_frame, text="BootStrap", command=self.BOOTChain)
        self.show_chart_btn.grid(row=3, column=0)

    def get_data_boot(self):
        table = Table(self.right_frame_top, dataframe=self.df_app_trian_BOOT, width=750, height=280)  # creates table with dataframe
        table.show()
            
    def show_all_BOOT(self,root):
            tk.Button(self.left_frame, text="Wyświetl dane BootStrap",
                  command=self.get_data_boot).grid(row=4, column=0)
    
    def get_stats(self):
        table = Table(self.right_frame_top, dataframe=self.df_app_trian_BOOT, width=750, height=280)
        table.show()
                
    def show_all_statistic(self,root):
            tk.Button(self.left_frame, text="Statystyki",
                  command=self.get_stats).grid(row=5, column=0)

    def enable_show_chart_btn(self, sender):
        self.Wylicza_statystyki()
        self.display_histogram()
            
    def updtcblist(self):
        list = self.products
        self.product_cb['values'] = list
     
    def show_product_combobox(self, root):
        l = tk.Label(self.left_frame,text = "Analizy dla poszczególnych lat")
        l.grid(row=6,column = 0  ,  pady = 10,sticky=tk.W)
        self.product_sv = tk.StringVar(value="")
        self.product_cb = ttk.Combobox(self.left_frame, textvariable=self.product_sv,postcommand = self.updtcblist)
        self.product_cb.set("Wybierz rok")
        self.product_cb.state = "readonly"
        self.product_cb.bind("<<ComboboxSelected>>",self.enable_show_chart_btn)
        self.product_cb.grid(row=7, column=0, padx=10, pady=10)

        
    def Wylicza_statystyki(self):
        self.selected_year = self.product_cb.get()
        desc = self.df_app_trian_BOOT[self.DICTio[self.selected_year]].describe(percentiles=[0.1,0.2,0.4,0.7,0.9,0.95,0.99,0.995], include='float')
        b = desc.index.to_list()
        self.right_chart_title_label.config(text = f"Tabela Statystyk dla roku {self.selected_year}")
        desc_df = pd.DataFrame({'Statystyka':b,
                              'Wartość':desc.to_list()})
        table = Table(self.right_frame_chart_right, dataframe=desc_df, width=150, height=280)  # creates table with dataframe
        table.show()
               
    def display_histogram(self): 
        f = Figure(figsize=(4,3), dpi=100)
        canvas = FigureCanvasTkAgg(f, master=self.right_frame_chart_left)
        self.left_chart_title_label.config(text=f"Histogram dla roku {self.selected_year}")
        canvas.get_tk_widget().grid(row=0, column=0,sticky = tk.W)
        p = f.gca()
        p.hist(self.df_app_trian_BOOT[self.DICTio[self.product_cb.get()]].to_list(), bins =20 )
        canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1150x750")
    root.title("Chainladder Bootstrap")
    app = AppChainladder(root)
    app.mainloop()