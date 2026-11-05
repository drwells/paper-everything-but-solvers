
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>

namespace Bench
{
  using namespace dealii;


  template <int dim>
  void
  test(const FiniteElement<dim> &fe, const unsigned int n_refine)
  {
    const MPI_Comm                            comm = MPI_COMM_WORLD;
    parallel::distributed::Triangulation<dim> tria(comm);

    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(comm) == 0);
    pcout << "Testing with " << fe.get_name() << std::endl;
    TimerOutput timer(pcout, TimerOutput::never, TimerOutput::wall_times);

    {
      TimerOutput::Scope scope(timer, "1_create_mesh");
      GridGenerator::hyper_cube(tria);
      tria.refine_global(n_refine);
      pcout << "Number of active cells: " << tria.n_global_active_cells()
            << std::endl;
    }
    DoFHandler<dim> dof_h(tria);
    {
      TimerOutput::Scope scope(timer, "2_distribute_dofs");
      dof_h.distribute_dofs(fe);
      pcout << "Number of degrees of freedom: " << dof_h.n_dofs() << std::endl;
    }

    LinearAlgebra::distributed::Vector<double> vector;
    {
      TimerOutput::Scope scope(timer, "3_create_vector");
      vector.reinit(dof_h.locally_owned_dofs(),
                    DoFTools::extract_locally_relevant_dofs(dof_h),
                    dof_h.get_mpi_communicator());
    }

    const unsigned int n_tests = 10;
    std::size_t        counter = 0;
    MPI_Barrier(comm);
    for (unsigned int t = 0; t < n_tests; ++t)
      {
        TimerOutput::Scope scope(timer, "4_get_dof_indices ");
        counter = 0;
        std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
        for (const auto &cell : dof_h.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(dof_indices);
              for (const auto a : dof_indices)
                counter += a;
            }
      }

    pcout << "All indices extracted sum " << counter << std::endl;

    MPI_Barrier(comm);
    counter = 0;
    for (unsigned int t = 0; t < n_tests; ++t)
      {
        TimerOutput::Scope scope(timer, "5_assemble_vector");
        vector = 0;

        // Use under-integration for the sake of benchmarking, since we use a
        // constant function '1' we do compute correctly anyway
        QGauss<dim>   quad(fe.degree);
        FEValues<dim> fe_values(fe, quad, update_values | update_JxW_values);
        std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
        for (const auto &cell : dof_h.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(dof_indices);
              fe_values.reinit(cell);
              for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                {
                  double sum = 0;
                  for (const unsigned int q :
                       fe_values.quadrature_point_indices())
                    sum += fe_values.shape_value(i, q) * fe_values.JxW(q);
                  vector(dof_indices[i]) += sum;
                }
            }
      }
    pcout << "Vector norm after assembly: " << vector.l2_norm() << std::endl;

    timer.print_wall_time_statistics(comm);

    pcout << std::endl << std::endl;
  }

} // namespace Bench


int
main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  for (unsigned int r = 7; r < 13; ++r)
    Bench::test<2>(dealii::FE_Q<2>(1), r);
  for (unsigned int r = 6; r < 12; ++r)
    Bench::test<2>(dealii::FE_Q<2>(2), r);
  for (unsigned int r = 6; r < 11; ++r)
    Bench::test<2>(dealii::FE_Q<2>(4), r);

  for (unsigned int r = 4; r < 9; ++r)
    Bench::test<3>(dealii::FE_Q<3>(1), r);
  for (unsigned int r = 3; r < 8; ++r)
    Bench::test<3>(dealii::FE_Q<3>(3), r);
  for (unsigned int r = 4; r < 9; ++r)
    Bench::test<3>(dealii::FE_DGQ<3>(1), r);
  for (unsigned int r = 3; r < 8; ++r)
    Bench::test<3>(dealii::FE_DGQ<3>(2), r);
}
