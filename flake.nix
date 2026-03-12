{
  description = "Statistical Volume Rendering Experiments";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        python = pkgs.python313;

        pyopenvdb = python.pkgs.buildPythonPackage rec {
          pname = "pyopenvdb";
          version = "12.1.0";
          format = "other";

          src = pkgs.fetchFromGitHub {
            owner = "AcademySoftwareFoundation";
            repo = "openvdb";
            tag = "v${version}";
            hash = "sha256-28vrIlruPl1tvw2JhjIAARtord45hqCqnA9UNnu4Z70=";
          };

          nativeBuildInputs = with pkgs; [
            cmake
            ninja
          ];

          buildInputs = with pkgs; [
            boost
            onetbb
            jemalloc
            c-blosc
            zlib
            openvdb
          ];

          propagatedBuildInputs = with python.pkgs; [
            numpy
            nanobind
          ];

          dontUsePipInstall = true;
          dontUseSetuptoolsBuild = true;

          cmakeFlags = [
            "-DOPENVDB_BUILD_CORE=OFF"
            "-DOPENVDB_BUILD_BINARIES=OFF"
            "-DOPENVDB_BUILD_PYTHON_MODULE=ON"
            "-DOPENVDB_PYTHON_WRAP_ALL_GRID_TYPES=ON"
            "-DUSE_NUMPY=ON"
            "-DOpenVDB_ROOT=${pkgs.openvdb}"
            "-Dnanobind_DIR=${python.pkgs.nanobind}/${python.sitePackages}/nanobind/cmake"
          ];

          buildPhase = ''
            cmake --build . --target openvdb_python
          '';

          installPhase = ''
            mkdir -p $out/${python.sitePackages}
            cp openvdb/openvdb/python/openvdb*.so $out/${python.sitePackages}/
          '';
        };

        slangpy = python.pkgs.buildPythonPackage rec {
          pname = "slangpy";
          version = "0.31.0";
          format = "wheel";

          src = python.pkgs.fetchPypi {
            inherit pname version;
            format = "wheel";
            dist = "cp313";
            python = "cp313";
            abi = "cp313";
            platform = "manylinux_2_34_x86_64";
            hash = "sha256-sIx871jat3SoyobTxam1FhdseyraXGd1FTT2202fCQM=";
          };

          nativeBuildInputs = with pkgs; [
            autoPatchelfHook
          ];

          buildInputs = with pkgs; [
            libx11
            vulkan-loader
            stdenv.cc.cc.lib
          ];

          propagatedBuildInputs = with python.pkgs; [
            numpy
          ];

          dontStrip = true;
        };

        pythonEnv = python.withPackages (ps: with ps; [
          matplotlib
          numpy
          slangpy
          opensimplex
          pyopenvdb
          torch
        ]);

        v3 = pkgs.stdenv.mkDerivation rec {
          pname = "v3";
          version = "0.5.2";
          src = pkgs.fetchsvn {
            url = "svn://svn.code.sf.net/p/volren/code/";
            rev = "1170";
            hash = "sha256-Qdauk3vElPMBcJKzcVW25W1+xmOJ7lNLTTl0HTWWfeE=";
          };
          sourceRoot = "${src.name}/viewer";
          nativeBuildInputs = [ pkgs.cmake ];
          buildInputs = [ pkgs.libGL pkgs.libGLU pkgs.libglut ];
          cmakeFlags = [
            "-DBUILD_VIEWER_APPS=OFF"
          ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonEnv
            shader-slang
            vulkan-headers
            vulkan-loader
            vulkan-tools
            vulkan-validation-layers
            shaderc
            glslang
            spirv-tools
            cmake
            ninja
            pkg-config
            v3
          ];

          shellHook = ''
            export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
            export VK_ICD_FILENAMES="/run/opengl-driver/share/vulkan/icd.d/intel_icd.x86_64.json"

            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
              pkgs.vulkan-loader
              pkgs.stdenv.cc.cc.lib
            ]}:$LD_LIBRARY_PATH"
          '';
        };
      }
    );
}
