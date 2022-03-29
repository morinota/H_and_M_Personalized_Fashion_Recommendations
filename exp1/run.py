from data import load_data
from preprocessing import read_data
def main():
    load_data()
    df, df_sub, dfu, dfi = read_data()



if __name__ == '__main__':
    main()