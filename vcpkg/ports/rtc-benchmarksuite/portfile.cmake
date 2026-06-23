vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO realtimechris/benchmarksuite
    REF "v${VERSION}"
    SHA512 4605c747482c18efebd2553167ede679f38eea92a1c82945438a8c49d099682ca0a393ebcebc656e1e5cc9cbaa57a201d4d437b0990b2b8951a5d2e49e139e49
    HEAD_REF main
)

set(VCPKG_BUILD_TYPE release)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/License.md")
