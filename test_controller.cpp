#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>
#include <string>
#include <map>

namespace py = pybind11;

class TestController {
private:
    int _ship_id = 0;

public:
    TestController() = default;

    std::tuple<float, float, bool, bool>
    actions(const std::map<std::string, py::object>& ship_state,
            const std::map<std::string, py::object>& game_state) {
        float thrust = 480.0f;
        float turn_rate = -9.0f;
        bool fire = true;
        bool drop_mine = false;
        return std::make_tuple(thrust, turn_rate, fire, drop_mine);
    }

    std::string name() const {
        return "Test Controller";
    }

    int ship_id() const {
        return _ship_id;
    }

    void set_ship_id(int value) {
        _ship_id = value;
    }
};

PYBIND11_MODULE(test_controller, m) {
    py::class_<TestController>(m, "TestController")
        .def(py::init<>())
        .def("actions", &TestController::actions)
        .def_property_readonly("name", &TestController::name)
        .def_property("ship_id", &TestController::ship_id, &TestController::set_ship_id);
}
