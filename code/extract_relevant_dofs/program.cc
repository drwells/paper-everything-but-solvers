
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

namespace Bench
{
  using namespace dealii;

  template <int dim, int spacedim>
  IndexSet
  extract_locally_relevant_dofs(const DoFHandler<dim, spacedim> &dof_handler)
  {
    // collect all the locally owned dofs
    IndexSet dof_set = dof_handler.locally_owned_dofs();

    std::vector<types::global_dof_index> dof_indices;
    std::vector<types::global_dof_index> dofs_on_ghosts;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_ghost())
      {
        dof_indices.resize(cell->get_fe().n_dofs_per_cell());
        cell->get_dof_indices(dof_indices);
        for (const auto dof_index : dof_indices)
          if (!dof_set.is_element(dof_index))
            dofs_on_ghosts.push_back(dof_index);
      }

    // sort and put into an index set
    std::sort(dofs_on_ghosts.begin(), dofs_on_ghosts.end());
    dof_set.add_indices(dofs_on_ghosts.begin(), dofs_on_ghosts.end());
    dof_set.compress();

    return dof_set;
  }

  template <int dim, int spacedim>
  IndexSet
  extract_locally_relevant_dofs_naive_vector(
    const DoFHandler<dim, spacedim> &dof_handler)
  {
    // collect all the locally owned dofs
    std::vector<types::global_dof_index> dof_indices;
    std::vector<types::global_dof_index> dofs_on_ghosts;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (!cell->is_artificial())
      {
        dof_indices.resize(cell->get_fe().n_dofs_per_cell());
        cell->get_dof_indices(dof_indices);
        for (const auto dof_index : dof_indices)
          dofs_on_ghosts.push_back(dof_index);
      }

    // sort and put into an index set
    std::sort(dofs_on_ghosts.begin(), dofs_on_ghosts.end());
    IndexSet dof_set(dof_handler.n_dofs());
    dof_set.add_indices(dofs_on_ghosts.begin(), dofs_on_ghosts.end());
    dof_set.compress();

    return dof_set;
  }

  template <int dim, int spacedim>
  IndexSet
  extract_locally_relevant_dofs_map(
    const DoFHandler<dim, spacedim> &dof_handler)
  {
    // collect all the locally owned dofs
    IndexSet dof_set = dof_handler.locally_owned_dofs();

    std::vector<types::global_dof_index> dof_indices;
    std::set<types::global_dof_index>    dofs_on_ghost;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_ghost())
      {
        dof_indices.resize(cell->get_fe().n_dofs_per_cell());
        cell->get_dof_indices(dof_indices);
        for (const auto dof_index : dof_indices)
          if (!dof_set.is_element(dof_index))
            dofs_on_ghost.insert(dof_index);
      }

    // sort and put into an index set
    for (const auto a : dofs_on_ghost)
      dof_set.add_index(a);
    dof_set.compress();

    return dof_set;
  }

  template <int dim, int spacedim>
  IndexSet
  extract_locally_relevant_dofs_unordered_map(
    const DoFHandler<dim, spacedim> &dof_handler)
  {
    // collect all the locally owned dofs
    IndexSet dof_set = dof_handler.locally_owned_dofs();

    std::vector<types::global_dof_index>        dof_indices;
    std::unordered_set<types::global_dof_index> dofs_on_ghost;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_ghost())
      {
        dof_indices.resize(cell->get_fe().n_dofs_per_cell());
        cell->get_dof_indices(dof_indices);
        for (const auto dof_index : dof_indices)
          if (!dof_set.is_element(dof_index))
            dofs_on_ghost.insert(dof_index);
      }

    dof_set.add_indices(dofs_on_ghost.begin(), dofs_on_ghost.end());
    dof_set.compress();

    return dof_set;
  }



  template <int dim>
  void
  test(const unsigned int degree, const unsigned int n_refine)
  {
    const MPI_Comm                            comm = MPI_COMM_WORLD;
    parallel::distributed::Triangulation<dim> tria(comm);

    FE_Q<dim> fe(degree);

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

    const unsigned int n_tests = 10;
    std::size_t        counter = 0;
    MPI_Barrier(comm);
    for (unsigned int t = 0; t < n_tests; ++t)
    {
      TimerOutput::Scope scope(timer, "3_dof_tools_relevant_dofs");
      const IndexSet     relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_h);
      counter +=
        relevant_dofs.n_elements() - dof_h.locally_owned_dofs().n_elements();
    }

    pcout << "DoFTools::extract_locally_relevant_dofs found "
          << Utilities::MPI::sum(counter, comm) / n_tests
          << " ghost dofs across all ranks" << std::endl;

    MPI_Barrier(comm);
    counter = 0;
    for (unsigned int t = 0; t < n_tests; ++t)
    {
      TimerOutput::Scope scope(timer, "4_vector_relevant_dofs");
      const IndexSet     relevant_dofs = extract_locally_relevant_dofs(dof_h);
      counter +=
        relevant_dofs.n_elements() - dof_h.locally_owned_dofs().n_elements();
    }

    pcout << "Manual extract_locally_relevant_dofs found "
          << Utilities::MPI::sum(counter, comm) / n_tests
          << " ghost dofs across all ranks" << std::endl;

    MPI_Barrier(comm);
    counter = 0;
    for (unsigned int t = 0; t < n_tests; ++t)
    {
      TimerOutput::Scope scope(timer, "5_naive_vector_relevant_dofs");
      const IndexSet     relevant_dofs =
        extract_locally_relevant_dofs_naive_vector(dof_h);
      counter +=
        relevant_dofs.n_elements() - dof_h.locally_owned_dofs().n_elements();
    }

    pcout << "Naive vector extract_locally_relevant_dofs found "
          << Utilities::MPI::sum(counter, comm) / n_tests
          << " ghost dofs across all ranks" << std::endl;

    MPI_Barrier(comm);
    counter = 0;
    for (unsigned int t = 0; t < n_tests; ++t)
    {
      TimerOutput::Scope scope(timer, "6_map_relevant_dofs");
      const IndexSet relevant_dofs = extract_locally_relevant_dofs_map(dof_h);
      counter +=
        relevant_dofs.n_elements() - dof_h.locally_owned_dofs().n_elements();
    }

    pcout << "Map-based extract_locally_relevant_dofs found "
          << Utilities::MPI::sum(counter, comm) / n_tests
          << " ghost dofs across all ranks" << std::endl;

    MPI_Barrier(comm);
    counter = 0;
    for (unsigned int t = 0; t < n_tests; ++t)
    {
      TimerOutput::Scope scope(timer, "7_unordered_set_relevant_dofs");
      const IndexSet     relevant_dofs =
        extract_locally_relevant_dofs_unordered_map(dof_h);
      counter +=
        relevant_dofs.n_elements() - dof_h.locally_owned_dofs().n_elements();
    }

    pcout << "unordered_map-based extract_locally_relevant_dofs found "
          << Utilities::MPI::sum(counter, comm) / n_tests
          << " ghost dofs across all ranks" << std::endl;

    timer.print_wall_time_statistics(comm);

    pcout << std::endl << std::endl;
  }

} // namespace Bench


int
main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  for (unsigned int r = 7; r < 13; ++r)
    Bench::test<2>(1, r);
  for (unsigned int r = 6; r < 12; ++r)
    Bench::test<2>(2, r);

  for (unsigned int r = 4; r < 9; ++r)
    Bench::test<3>(1, r);
  for (unsigned int r = 3; r < 8; ++r)
    Bench::test<3>(2, r);
}
