#include "boost/python.hpp"
#include "boost/python/suite/indexing/vector_indexing_suite.hpp"
#include "boost/python/numpy.hpp"
#include "boost/python/stl_iterator.hpp"
#include <vector>
#include "AIC.hpp"
#include "SF.hpp"

using namespace boost::python;
using namespace IC;

using namespace boost::python::numpy;

/// @brief Type that allows for registration of conversions from
///        python iterable types.
struct iterable_converter
{
  /// @note Registers converter from a python interable type to the
  ///       provided type.
  template <typename Container>
  iterable_converter&
  from_python()
  {
    boost::python::converter::registry::push_back(
      &iterable_converter::convertible,
      &iterable_converter::construct<Container>,
      boost::python::type_id<Container>());

    // Support chaining.
    return *this;
  }

  /// @brief Check if PyObject is iterable.
  static void* convertible(PyObject* object)
  {
    return PyObject_GetIter(object) ? object : NULL;
  }

  /// @brief Convert iterable PyObject to C++ container type.
  ///
  /// Container Concept requirements:
  ///
  ///   * Container::value_type is CopyConstructable.
  ///   * Container can be constructed and populated with two iterators.
  ///     I.e. Container(begin, end)
  template <typename Container>
  static void construct(
    PyObject* object,
    boost::python::converter::rvalue_from_python_stage1_data* data)
  {
    namespace python = boost::python;
    // Object is a borrowed reference, so create a handle indicting it is
    // borrowed for proper reference counting.
    python::handle<> handle(python::borrowed(object));

    // Obtain a handle to the memory block that the converter has allocated
    // for the C++ type.
    typedef python::converter::rvalue_from_python_storage<Container>
                                                                storage_type;
    void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

    typedef python::stl_input_iterator<typename Container::value_type>
                                                                    iterator;

    // Allocate the C++ type into the converter's memory block, and assign
    // its handle to the converter's convertible variable.  The C++
    // container is populated by passing the begin and end iterators of
    // the python object to the container's constructor.
    new (storage) Container(
      iterator(python::object(handle)), // begin
      iterator());                      // end
    data->convertible = storage;
  }
};

struct SF_callback : SF, wrapper<SF>
{
    double operator() (const vector<size_t> &B) const
    {
        return this->get_override("__call__")();
    }
    size_t size() const
    {
        return this->get_override("size")();
    }
};



void test1(std::vector<double> values)
{
  for (auto&& value: values)
    std::cout << value << std::endl;
}

void test2(std::vector<vector<double>> values)
{
  for (auto&& value: values)
    for (auto&& value0: value)
      std::cout << value0 << std::endl;
}

void test3(std::vector<size_t> values)
{
  for (auto&& value: values)
    std::cout << value << std::endl;
}

BOOST_PYTHON_MODULE(AIC)
{
    iterable_converter()
        .from_python<vector<double> >()
        .from_python<vector<vector<double> > >()
        .from_python<vector<size_t> >()
        .from_python<vector<vector<size_t> > >()
    ;
    class_<std::vector<double>>("D_V").def(vector_indexing_suite<std::vector<double>>());
    class_<std::vector<size_t>>("UI_V").def(vector_indexing_suite<std::vector<size_t>>());
    class_<std::vector<std::vector<double>>>("D_VV").def(vector_indexing_suite<std::vector<std::vector<double>>>());
    class_<std::vector<std::vector<size_t>>>("UI_VV").def(vector_indexing_suite<std::vector<std::vector<size_t>>>());

    class_<HC>("HC", init<size_t>())
        .def("getPartition", &HC::getPartition)
        .def("similarity", &HC::similarity)
        .def("getCriticalValues", &HC::getCriticalValues)
    ;

    class_<SF_callback, boost::noncopyable>("SF", no_init)
        .def("__call__", pure_virtual(&SF::operator()))
        .def("size", pure_virtual(&SF::size))
    ;

    bool (AIC_TE::*agg1)(void) = &AIC_TE::agglomerate;
    bool (AIC_TE::*agg2)(double, double) = &AIC_TE::agglomerate;
    class_<AIC_TE>("AIC_TE", init<vector<double>, int>())
        .def("agglomerate", agg1)
        .def("agglomerate", agg2)
        .def("getPartition", &AIC_TE::getPartition)
        .def("getCriticalValues", &AIC_TE::getCriticalValues)
    ;

    class_<TableEntropy, bases<SF>>("TableEntropy", init<vector<double>, int>())
        .def("subsetIndex", &TableEntropy::subsetIndex)
        .def("subsetVector", &TableEntropy::subsetVector)
        .staticmethod("subsetVector")
        .staticmethod("subsetIndex")
    ;

    class_<CL, bases<HC>>("CL", init<size_t, vector<size_t>, vector<size_t>, vector<double>>());

    // class_<HC>("HC", init<size_t>())
    // .def("getPartition", &HC::getPartition)
    // .def("similarity", &HC::similarity)
    // .def("getCriticalValues", &HC::getCriticalValues)
    // .def("merge",&HC::merge)
    // .def("find",&HC::find);
    def("test1", &test1);
    def("test2", &test2);
    def("test3", &test3);

}