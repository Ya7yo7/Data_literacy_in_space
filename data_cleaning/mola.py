import rioxarray as rxr
import rasterio
from rasterio.plot import show

with rasterio.open("data_files/mola_data/mega90n000cb.img", driver="PDS4") as src:
    show(src)


# Try opening with ENVI driver (common for MOLA data)
mola = rxr.open_rasterio("data_files/mola_data/mega90n000cb.img", masked=True, driver='ENVI')
print(mola)
mola.plot()
