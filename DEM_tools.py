import subprocess

def hillshade(input_path, output_path, sun_angle=315, sun_altitude=45):
    arg = f"gdaldem hillshade -az {sun_angle} -alt {sun_altitude} {input_path} {output_path}"
    arg = arg.split(" ")
    subprocess.run(arg)


def slope(input_path, output_path):
    arg = f"gdaldem slope {input_path} {output_path}"
    arg = arg.split(" ")
    subprocess.run(arg)


def aspect(input_path, output_path):
    arg = f"gdaldem aspect {input_path} {output_path}"
    arg = arg.split(" ")
    subprocess.run(arg)


def TRI(input_path, output_path):
    arg = f"gdaldem TRI {input_path} {output_path}"
    arg = arg.split(" ")
    subprocess.run(arg)


def TPI(input_path, output_path):
    arg = f"gdaldem TPI {input_path} {output_path}"
    arg = arg.split(" ")
    subprocess.run(arg)


def roughness(input_path, output_path):
    arg = f"gdaldem roughness {input_path} {output_path}"
    arg = arg.split(" ")
    subprocess.run(arg)
    
    
if __name__ == "__main__":
    main_path = "C:/Users/Tim/Desktop/Myna/mine_applications/DEM/"
    hillshade(main_path + "NZDEM_SoS_v1-0_10_Napier_gf.tif", main_path + "hillshade.tif")
    slope(main_path + "NZDEM_SoS_v1-0_10_Napier_gf.tif", main_path + "slope.tif")
    aspect(main_path + "NZDEM_SoS_v1-0_10_Napier_gf.tif", main_path + "aspect.tif")
    TRI(main_path + "NZDEM_SoS_v1-0_10_Napier_gf.tif", main_path + "TRI.tif")
    roughness(main_path + "NZDEM_SoS_v1-0_10_Napier_gf.tif", main_path + "roughness.tif")
    TPI(main_path + "NZDEM_SoS_v1-0_10_Napier_gf.tif", main_path + "TPI.tif")
