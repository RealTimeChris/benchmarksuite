vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO realtimechris/benchmarksuite
    REF "v${VERSION}"
    SHA512 f3cc79156b957a9ab798468bf16aefb75c735e7093540d74ef67b0b2c4ef10242540e0990c192837e49295dcb11f94785540ba0b6481c825e13205fed7d6ecad
    HEAD_REF main
)

set(VCPKG_BUILD_TYPE release)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/License.md")
