{
  description = "Statistical Volume Rendering Experiments";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
    nixgl.url = "github:nix-community/nixGL";
  };

  outputs = { self, nixpkgs, flake-utils, nixgl }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            config.cudaSupport = true;
            config.cudaVersion = "13";
          };
        };

        nixglPinned = pkgs.callPackage (nixgl.outPath + "/default.nix") {
          nvidiaVersion = "580.142";
          nvidiaHash = "sha256-IJFfzz/+icNVDPk7YKBKKFRTFQ2S4kaOGRGkNiBEdWM=";
        };

        python = pkgs.python313;

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
            typing-extensions
          ];

          dontStrip = true;
        };

        accelerate = python.pkgs.accelerate.override {
          torch = python.pkgs.torch-bin;
        };

        pythonEnv = python.withPackages (ps: with ps; [
          matplotlib
          numpy
          slangpy
          opensimplex
          torch-bin
          torchvision-bin
          typer
          diffusers
          einops
          accelerate
          scipy
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
            nixglPinned.nixVulkanNvidia
            renderdoc
            cudatoolkit
            cudaPackages.nsight_compute
            cudaPackages.nsight_systems
          ];

          shellHook = ''
            export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
            export VK_ICD_FILENAMES="/run/opengl-driver/share/vulkan/icd.d/intel_icd.x86_64.json"

            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
              pkgs.vulkan-loader
              pkgs.stdenv.cc.cc.lib
            ]}:$LD_LIBRARY_PATH"

            export CUDA_PATH=${pkgs.cudatoolkit}
          '';
        };
      }
    );
}
