import time

from arguments import get_args
from AlphaJoin_1.supervised import supervised

if __name__ == '__main__':
    args = get_args()

    # print(args)

    trainer = supervised(args)
    print("Pretreatment running...")
    start = time.clock()
    trainer.pretreatment("runtime_laptop")
    elapsed = (time.clock() - start)
    print("Pretreatment time used:", elapsed)
