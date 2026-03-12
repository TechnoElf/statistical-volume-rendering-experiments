# PVM File Format Specification

## Overview

PVM (Portable Volume Map) is a simple, human-readable file format for storing volumetric data such as medical CT or MRI scans. It is the native format used by the Versatile Volume Viewer (V³).

The format is designed to be:
- Simple and human-readable (ASCII header)
- Efficient for 3D volume storage
- Extensible with metadata support
- Optionally compressed using DDS compression

## File Structure

A PVM file consists of two parts:
1. **ASCII Header** - Contains metadata about the volume
2. **Binary Data** - Raw voxel data

## Format Versions

There are three versions of the PVM format:

### PVM (Version 1)
Basic format with uniform voxel spacing (assumed to be 1.0 in all dimensions).

### PVM2 (Version 2)
Adds support for non-uniform voxel spacing.

### PVM3 (Version 3)
Extends PVM2 with additional metadata fields for volume description, courtesy information, scan parameters, and comments.

## Header Format

### PVM Version 1

```
PVM
<WIDTH> <HEIGHT> <DEPTH>
<COMPONENTS>
<binary data>
```

**Fields:**
- `PVM` - Magic identifier (no version number)
- `<WIDTH>` - Volume width in voxels (unsigned integer)
- `<HEIGHT>` - Volume height in voxels (unsigned integer)
- `<DEPTH>` - Volume depth in voxels (unsigned integer)
- `<COMPONENTS>` - Number of components per voxel (see Components section)

**Example:**
```
PVM
256 256 256
1
<binary data>
```

### PVM2 Version 2

```
PVM2
<WIDTH> <HEIGHT> <DEPTH>
<VOXEL_WIDTH> <VOXEL_HEIGHT> <VOXEL_DEPTH>
<COMPONENTS>
<binary data>
```

**Additional Fields:**
- `<VOXEL_WIDTH>` - Voxel spacing in X dimension (float)
- `<VOXEL_HEIGHT>` - Voxel spacing in Y dimension (float)
- `<VOXEL_DEPTH>` - Voxel spacing in Z dimension (float)

**Example:**
```
PVM2
256 256 256
1 1 1.5
1
<binary data>
```

### PVM3 Version 3

```
PVM3
<WIDTH> <HEIGHT> <DEPTH>
<VOXEL_WIDTH> <VOXEL_HEIGHT> <VOXEL_DEPTH>
<COMPONENTS>
<binary data>
<description string>\0
<courtesy string>\0
<parameter string>\0
<comment string>\0
```

**Additional Fields (after binary data):**
- `<description>` - Object description (null-terminated string)
- `<courtesy>` - Courtesy information: institution, copyright, contact, etc. (null-terminated string)
- `<parameter>` - Scan parameters: date, time, mode, etc. (null-terminated string)
- `<comment>` - Additional comments (null-terminated string)

Any of these strings can be empty (just a null byte). If all metadata strings are empty, the format falls back to PVM2.

**Example:**
```
PVM3
256 256 256
1 1 1.5
1
<binary data>
Brain MRI scan\0
Medical University, 2020\0
Date: 2020-01-15, Scanner: Siemens\0
High resolution scan\0
```

## Components

The `<COMPONENTS>` field specifies the data type and number of components per voxel:

| Value | Description | Bytes per Voxel |
|-------|-------------|-----------------|
| 1 | 8-bit unsigned byte | 1 |
| 2 | 16-bit signed short (MSB order) | 2 |
| 3 | RGB (3 × 8-bit) | 3 |
| 4 | RGBA (4 × 8-bit) | 4 |

### Note on 16-bit Data

- 16-bit data is stored as **signed short** values in **MSB (Most Significant Byte) order**
- Maximum value is 32767 (not 65535)
- If unsigned short data with values > 32767 is encountered, it is automatically converted by dividing by 2

## Binary Data Layout

The binary data immediately follows the header. The data is stored in **row-major order**:

```
for z = 0 to DEPTH-1:
    for y = 0 to HEIGHT-1:
        for x = 0 to WIDTH-1:
            write voxel(x, y, z)
```

Total size of binary data: `WIDTH × HEIGHT × DEPTH × COMPONENTS` bytes

## Optional Comments (PVM Version 1 only)

PVM Version 1 supports optional comment lines in the header:

```
PVM
# This is a comment
# Comments start with # and end with newline
256 256 256
1
<binary data>
```

Comments are preserved when reading and writing PVM files.

## DDS Compression

PVM files can be optionally compressed using DDS (Differential Data Stream) compression. Despite the name similarity to DirectDraw Surface, this is a custom lossless compression format designed specifically for volumetric data.

### DDS Format Versions

There are two DDS wrapper versions:

- **DDS v3d** - For files up to 16 MB (16,777,216 bytes)
- **DDS v3e** - For files larger than 16 MB

### File Structure

Compressed PVM files have this structure:

```
<DDS_MAGIC>
<compressed data>
```

Where `<DDS_MAGIC>` is either:
- `"DDS v3d\n"` (8 bytes) for v3d format
- `"DDS v3e\n"` (8 bytes) for v3e format

The compressed data contains the entire PVM file (header + binary data + metadata).

### Compression Algorithm

DDS compression uses a differential encoding scheme with run-length encoding:

#### 1. **Deinterleaving**
   - Data is deinterleaved based on component count (skip parameter)
   - For multi-byte data (e.g., 16-bit), bytes are reorganized to group similar byte positions together
   - This improves compression by grouping similar values

#### 2. **Differential Encoding**
   - Each byte is encoded as a difference from previous values
   - Two reference points are used:
     - Previous value in the stream
     - Value from previous "strip" (scanline/row)
   - Formula: `delta = current - previous - strip_previous + strip_previous_previous`
   - Deltas are clamped to range [-128, 127]

#### 3. **Bit-Packing with Run-Length Encoding**
   - Consecutive deltas requiring the same number of bits are grouped
   - Each group is encoded as:
     - **Run length** (7 bits): Number of values in the group (1-127)
     - **Bit count** (3 bits): Number of bits needed per value (0-7, encoded)
     - **Delta values**: Variable-bit-width encoded deltas
   - This creates efficient compression for smooth gradients and uniform regions

#### 4. **Interleaving Control**
   - For files > 16 MB (DDS v3e), data is processed in 16 MB chunks
   - Each chunk is deinterleaved/interleaved independently
   - This prevents memory issues with very large volumes

### Compression Parameters

When writing DDS files, two parameters control the compression:

- **skip** (1-4 bytes): Component size for deinterleaving
  - 1 = 8-bit data
  - 2 = 16-bit data  
  - 3 = RGB data
  - 4 = RGBA data

- **strip** (1-65536): Strip width for differential prediction
  - Typically set to image width for 2D slices
  - Enables better prediction along scanlines
  - Default is 1 (no strip-based prediction)

### Reading Compressed PVM Files

The read process is transparent:

1. Attempt to read as DDS file (check for magic identifier)
2. If DDS magic found, decompress the data
3. Parse the decompressed PVM header and data
4. If not DDS, read as uncompressed PVM

### Writing Compressed PVM Files

PVM files are automatically DDS-compressed when:

- Writing PVM3 format (with metadata)
- Explicitly using `writeDDSfile()` function

### Compression Efficiency

DDS compression is particularly effective for:

- Medical imaging data (CT/MRI scans with smooth gradients)
- Volumetric terrain data
- Data with spatial coherence
- 16-bit volumes (often compress to 30-50% of original size)

Typical compression ratios:
- **8-bit medical data**: 40-60% of original
- **16-bit medical data**: 30-50% of original  
- **Noisy data**: 70-90% of original
- **Uniform regions**: Can compress to < 10% of original

### Tools

- **dds** tool: Compress/decompress any file with DDS
  - `dds file.pvm` - If uncompressed, compresses it; if compressed, decompresses it
  - Works in-place (overwrites the input file)

- **pvmdds** tool: Similar functionality for PVM files specifically

### Implementation Notes

- DDS compression is **lossless** - original data is perfectly reconstructed
- Compression is CPU-intensive but decompression is fast
- Uses bit-level packing for maximum efficiency
- Handles both little-endian and big-endian architectures
- Memory efficient: processes data in 1 MB blocks during compression

### Example

To manually compress a PVM file:

```bash
# Compress a PVM file
dds uncompressed.pvm  # Creates compressed version

# Decompress it back
dds compressed.pvm    # Restores uncompressed version
```

The V³ viewer handles compressed PVM files transparently - no manual decompression needed.

## Coordinate Systems and Geographic Metadata

PVM files can include geographic grid descriptors through the PNM comment mechanism. This is primarily used for terrain data. The descriptor includes:

- Coordinate system (Lat/Lon or UTM)
- Corner coordinates (SW, NW, NE, SE)
- Cell size
- Vertical scaling
- Datum and zone information
- Missing data value

This metadata is stored in structured comment format starting with the magic string `BOX` or `DEM` or `TEX`.

## Reading PVM Files

### Algorithm

1. Read magic identifier (4-5 bytes)
2. Determine version:
   - `PVM\n` → Version 1
   - `PVM2\n` → Version 2
   - `PVM3\n` → Version 3
   - `DDS v3d` → Compressed (decompress first)
3. Skip any comment lines (Version 1 only)
4. Read dimensions (WIDTH, HEIGHT, DEPTH)
5. Read voxel spacing (Version 2 and 3 only)
6. Read component count
7. Allocate buffer: `WIDTH × HEIGHT × DEPTH × COMPONENTS` bytes
8. Read binary data
9. Read metadata strings (Version 3 only)

## Writing PVM Files

### Version Selection

- Use **PVM** (v1) if: `scalex == scaley == scalez == 1.0` and no metadata
- Use **PVM2** if: non-uniform spacing and no metadata
- Use **PVM3** if: any metadata is present

### Algorithm

1. Write magic identifier based on version
2. Write dimensions
3. Write voxel spacing (Version 2 and 3)
4. Write component count
5. Write binary data
6. Write metadata strings as null-terminated (Version 3 only)
7. Optionally compress with DDS

## Tools

The following command-line tools are available for working with PVM files:

- `raw2pvm` - Convert raw volume data to PVM format
- `pvm2raw` - Convert PVM to raw volume data
- `pvm2pgm` - Convert PVM to a stack of PGM images
- `pgm2pvm` - Convert a stack of PGM images to PVM
- `pvm2pvm` - Quantize 16-bit PVM to 8-bit using non-linear mapping
- `pvminfo` - Display PVM file information and add/modify metadata
- `pvmplay` - Cross-section viewer for PVM files
- `pvmdds` - Work with DDS compressed PVM files

## Data Conversion Notes

### From Raw to PVM

The `raw2pvm` tool supports various input formats:

- **1** - 8-bit unsigned
- **2** - 16-bit MSB (signed)
- **-2** - 16-bit LSB (converted to MSB)
- **-32768** - 16-bit LSB signed (converted to MSB)
- **32767** - 16-bit MSB signed
- **4** - IEEE float (converted to 16-bit)
- **3** - RGB
- **-3** - RGB intensity (converted to grayscale)

### From PVM to Raw

Preserves the native format with proper byte ordering.

## Limitations

- Maximum dimensions: No explicit limit (depends on system memory)
- Supported data types: 8-bit, 16-bit, RGB, RGBA only
- No built-in support for floating-point data (must be converted)
- No built-in compression (except via DDS wrapper)

## Compatibility

The PVM format is compatible with:
- V³ (Versatile Volume Viewer)
- QtV3 (Qt-based volume viewer)
- libMini terrain rendering library
- Standard medical imaging workflows (with DICOM conversion)

## File Extension

The standard file extension is `.pvm`

Compressed files may use `.pvm` (transparently handled) or sometimes appear with DDS-related extensions, though this is not standard.

## Example Code

### Writing a PVM File (C/C++)

```c
#include "pnmbase.h"

unsigned char *volume = ...; // your volume data
unsigned int width = 256, height = 256, depth = 256;
unsigned int components = 1; // 8-bit
float scalex = 1.0f, scaley = 1.0f, scalez = 1.5f;

writePVMvolume("output.pvm", volume, 
               width, height, depth, components,
               scalex, scaley, scalez);
```

### Reading a PVM File (C/C++)

```c
#include "pnmbase.h"

unsigned int width, height, depth, components;
float scalex, scaley, scalez;

unsigned char *volume = readPVMvolume("input.pvm",
                                      &width, &height, &depth, &components,
                                      &scalex, &scaley, &scalez);

if (volume != NULL) {
    // Process volume data
    free(volume);
}
```

### Writing PVM3 with Metadata (C/C++)

```c
#include "pnmbase.h"

unsigned char *description = (unsigned char *)"Brain MRI";
unsigned char *courtesy = (unsigned char *)"Medical Center 2020";
unsigned char *parameter = (unsigned char *)"Date: 2020-01-15";
unsigned char *comment = (unsigned char *)"High resolution";

writePVMvolume("output.pvm", volume,
               width, height, depth, components,
               scalex, scaley, scalez,
               description, courtesy, parameter, comment);
```

## References

- V³ (Versatile Volume Viewer) - http://stereofx.org
- Author: Stefan Roettger
- License: GPL 2+ (LGPL 2.1 for library components)

## Revision History

This specification is based on the V³ VIEWER-5.2 source code analysis.

- **Version 1.0** - Original PVM format
- **Version 2.0** - Added voxel spacing support
- **Version 3.0** - Added metadata support
