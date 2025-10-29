from data import create_wikitext2_dataset

if __name__ == "__main__":

    train, eval = create_wikitext2_dataset()
    print(len(train))
    print(len(eval))
